//! Analytic stress of the k-point closed-shell RKS energy for META-GGA
//! functionals (SCAN, r2SCAN, TPSS; `ferric_pbc::kxcstress`, tau strain term).
//!
//! Derivation (the new piece). At fixed `D`, `tau = (1/2N_k) sum_k sum_i sum_{mu nu}
//! D_{mu nu} d_i phi_mu (d_i phi_nu)^*`, and `d(d_i phi)/d eps_ab = m2[3i+a, b]`
//! (the same strained-gradient moments the GGA `grad rho` strain already uses),
//! so `d tau/d eps_ab = (1/N_k) Re sum_{mu,i} m2[3i+a,b]_mu md^i_mu` (the 1/2
//! cancels the mu<->nu doubling), `md^i_mu = sum_nu D_{mu nu} (d_i phi_nu)^*`.
//! The stress gains `sum_g w_g v_tau d tau/d eps`; the weight response already
//! carries `eps_g = rho eps_xc(rho, sigma, tau)`; and `V^tau` enters the
//! Pulay Fock (guarded against the SCF's own Fock).
//!
//! Anchors (artifact hypothesis stated at each test):
//! * `null_tau_path_reproduces_the_gga_stress_exactly` -- PBE through the tau
//!   path (`v_tau = 0`) is the PBE stress;
//! * `fixed_density_xc_stress_matches_fd_of_exc` -- `xc_ao + xc_weight` against
//!   the central FD of `E_xc[D]` at FIXED `D` under cell strain (no SCF, the
//!   independent construction), SCAN and r2SCAN, atom grid, all nine
//!   components; the same comparison with each XC strain defect injected must
//!   fail; `fixed_density_xc_stress_matches_fd_on_the_uniform_grid` repeats it
//!   on the uniform grid with the same defects;
//! * `p_shell_triclinic_fixed_density_xc_stress_matches_fd` -- s+p basis on the
//!   triclinic cell, all nine components, and the `TauTranspose` defect (the
//!   (a,b)-transposed tau strain term) that the cubic s-only fixtures cannot see;
//! * `uniform_grid_stress_matches_pyscf_fftdf_krks_stress_scan` -- PySCF 2.13.1
//!   analytic k-point SCAN stress (JSON under testdata/reference);
//! * `meta_gga_stress_matches_fd_of_the_scf_energy` -- total dE/d eps against FD
//!   of the converged SCF energy (uniform grid, three components);
//! * `supercell_identity_per_part` -- the 1x1x3 stress equals the explicit
//!   3-cell supercell at 1x1x1 (same code; checks the k sum and phases);
//! * RS-GDF J: `rsgdf_meta_gga_stress_matches_fd_of_the_scf_energy` (FD of the
//!   SCAN RS-GDF SCF energy, tau defects on that route must miss; all nine
//!   components SCAN + r2SCAN ignored) and
//!   `rsgdf_minus_dense_stress_transfers_from_pbe_to_scan` (the route
//!   difference is the fit error, the same for SCAN as for PBE);
//! * refusals, the Fock guard, production-wired mutants;
//! * TPSS (issue #492): `uniform_grid_stress_matches_pyscf_fftdf_krks_stress_tpss`
//!   (PySCF `krks_stress` accepts TPSS) and
//!   `tpss_stress_matches_fd_of_the_scf_energy` (FD of the TPSS SCF energy,
//!   with the tau mutants); the H4 triclinic cell and the TPSS k-mesh ==
//!   supercell identity are weekly (`validation_pbc_mgga_derivs.rs`).
//!
//! NOT covered: hybrid meta-GGAs (none resolve), range-separated / VV10 /
//! double hybrid meta-GGAs (refused), open-shell meta-GGA stress (UKS/ROKS
//! keep their typed refusal; open-shell TPSS is refused), TPSS on RS-GDF (refused by the bindings), the
//! range-split RS-GDF build, d shells, ECPs, heavy atoms, metals. The
//! Gamma-point meta-GGA stress is this code on the 1x1x1 mesh
//! (`pbc_gamma_mgga_derivs.rs`).
//!
//! WEEKLY HALF: the four anchors whose CI time is >= 120 s (the SCAN /
//! r2SCAN fixed-density XC FD on the atom and uniform grids, all nine
//! components with the XC mutants, and the SCAN / TPSS PySCF 61^3 stress
//! pins) run in `validation_pbc_kxcstress_mgga.rs`; the helpers are
//! `kxcstress_mgga_support/mod.rs`. Per-commit canaries for them:
//! `tpss_components_xc_stress_matches_fd_of_exc` (fixed-D XC FD, atom grid,
//! all nine), `uniform_grid_fixed_density_xc_stress_canary` (SCAN, uniform
//! grid, two components, every XC mutant),
//! `uniform_grid_fixed_density_xc_stress_r2scan_canary` (r2SCAN, uniform
//! grid, zz, the two tau-strain mutants), `mutants_are_caught`, and the
//! SCF-energy FD anchors (`meta_gga_stress_matches_fd_of_the_scf_energy`,
//! `tpss_stress_matches_fd_of_the_scf_energy`). The TPSS libxc name-table
//! mutant the TPSS PySCF stress pin exists for is also caught per commit by
//! the TPSS energy pin `pbc_krks.rs::tpss_krks_and_gamma_match_pinned_pyscf`.

mod common;
#[path = "kxcstress_mgga_support/mod.rs"]
mod support;

use common::*;
use support::*;

/// Null-tau limit. Artifact hypothesis: a tau path that perturbed the non-tau
/// terms (mis-sized plane, stray accumulation) would shift the PBE stress;
/// physics predicts exactly 0 because `v_tau = 0`.
#[test]
fn null_tau_path_reproduces_the_gga_stress_exactly() {
    let fx = fixture(
        &H2,
        [1, 1, 3],
        "PBE",
        Jk::Dense,
        atom_grid(20, 50, Some(CUTOFF)),
    );
    let base = stress(&fx, None);
    let hooked = stress(&fx, Some(KXcStressMutation::StressTauPath));
    let d = mdiff(&hooked.de_deps, &base.de_deps);
    eprintln!("PBE through the tau path: max diff {d:.2e}");
    assert!(d < 1e-12, "{d:e}");
}

/// TPSS through its raw component names on a PBE density (`GuardSkippedOnly`
/// skips only the Fock guard, which the XC parts do not read).
#[test]
fn tpss_components_xc_stress_matches_fd_of_exc() {
    let grid = atom_grid(20, 50, Some(CUTOFF));
    let fx = fixture(&H2, [1, 1, 3], "PBE", Jk::Dense, grid.clone());
    let d0 = fx.res.scf.densities.clone();
    for f in ["MGGA_X_TPSS", "MGGA_C_TPSS"] {
        let mut fx2 = fixture_cell(fx.cell.clone(), [1, 1, 3], "PBE", Jk::Dense, grid.clone());
        fx2.functional = f.to_string();
        let s = stress(&fx2, Some(KXcStressMutation::GuardSkippedOnly));
        let fd = fd_xc_all9(&fx.cell, [1, 1, 3], f, &grid, &d0, FD_H_XC);
        let miss = mdiff(&xc_part(&s), &fd);
        eprintln!(
            "{f}: |xc - FD| {miss:.2e}; |xc| max {:.3e}",
            mmax(&xc_part(&s))
        );
        assert!(miss < FD_BAR_XC, "{f}: {miss:e}");
    }
}

/// Per-commit canary of the weekly
/// `fixed_density_xc_stress_matches_fd_on_the_uniform_grid`
/// (`validation_pbc_kxcstress_mgga.rs`): the same uniform-grid fixed-density
/// XC FD and the same XC strain defects, SCAN only, on the zz and xz strains
/// (the bond axis and one shear; the weekly test runs all nine for SCAN and
/// r2SCAN). Measured 81 s locally on a loaded box (2026-10-30).
#[test]
fn uniform_grid_fixed_density_xc_stress_canary() {
    uniform_fixed_density_xc_case(&["SCAN"], &[(2, 2), (0, 2)], &XC_MUTANTS);
}

/// The r2SCAN row of the same canary, as its own test (in one test with the
/// SCAN row it took 134 s locally, over the 120 s per-commit line; the
/// r2SCAN SCF fixture is most of the extra 53 s): zz strain, the two
/// tau-strain mutants (`NoTauStrain`, `TauDoubled`). It runs r2SCAN's tau
/// strain path per commit. The FD is of the same name, so a name mis-route
/// (r2SCAN -> SCAN) is NOT visible here; the CLI force canary
/// `kpoint_uks_r2scan_force_at_the_input_geometry_differs_from_scan` and
/// `pbc_krks.rs::gamma_mgga_matches_pinned_pyscf_gamma_rks` see it.
#[test]
fn uniform_grid_fixed_density_xc_stress_r2scan_canary() {
    uniform_fixed_density_xc_case(
        &["R2SCAN"],
        &[(2, 2)],
        &[
            KXcStressMutation::NoTauStrain,
            KXcStressMutation::TauDoubled,
        ],
    );
}

// ------------------------------------------------------------ SCF-energy FD

/// Total dE/d eps against FD of the converged SCAN SCF energy, uniform grid.
#[test]
fn meta_gga_stress_matches_fd_of_the_scf_energy() {
    let grid = KXcGrid::Uniform([24, 24, 24]);
    let f = "SCAN";
    let fx = fixture(&H2, [1, 1, 3], f, Jk::Dense, grid.clone());
    let s = stress(&fx, None);
    let h = 1e-4;
    let mut worst = 0.0_f64;
    for (i, j) in [(0, 0), (2, 2), (0, 2)] {
        let ep = energy_on(&strained(&fx.cell, i, j, h), [1, 1, 3], f, Jk::Dense, &grid);
        let em = energy_on(
            &strained(&fx.cell, i, j, -h),
            [1, 1, 3],
            f,
            Jk::Dense,
            &grid,
        );
        let fd = (ep - em) / (2.0 * h);
        worst = worst.max((s.de_deps[i][j] - fd).abs());
    }
    eprintln!("SCAN uniform 24^3: |analytic - FD of SCF energy| {worst:.2e}");
    assert!(worst < 5e-7, "{worst:e}");
}

/// The 1x1x3 stress per cell equals the explicit 3-cell supercell at 1x1x1 by
/// the same code (carries the Bloch phases and the 1/N_k; not the tau
/// derivative itself). Artifact hypothesis: a missing phase / 1/N_k in the tau
/// term would break it at the size of the tau term, not at 1e-10.
#[test]
fn supercell_identity_per_part() {
    let grid = atom_grid(20, 50, Some(CUTOFF));
    let f = "SCAN";
    let k = stress(&fixture(&H2, [1, 1, 3], f, Jk::Dense, grid.clone()), None);
    let sc = supercell_pos(&H2, 4.0, [1, 1, 3]);
    let g = stress(
        &fixture_cell(sc, [1, 1, 1], f, Jk::Dense, grid.clone()),
        None,
    );
    let d_ao = mdiff(&mscale(&k.xc_ao, 3.0), &g.xc_ao);
    let d_w = mdiff(&mscale(&k.xc_weight, 3.0), &g.xc_weight);
    let d_t = mdiff(&mscale(&k.de_deps, 3.0), &g.de_deps);
    eprintln!("SCAN 1x1x3 vs supercell: xc_ao {d_ao:.2e} xc_weight {d_w:.2e} total {d_t:.2e}");
    assert!(d_ao < 1e-9 && d_w < 1e-9 && d_t < 1e-9);
}

// ----------------------------------------------------- refusals, guard, mutants

#[test]
fn refusals_and_the_fock_guard() {
    let fx = fixture(
        &H2,
        [1, 1, 3],
        "SCAN",
        Jk::Dense,
        atom_grid(20, 50, Some(CUTOFF)),
    );
    let eri = KDenseAftEri::build(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &fx.hk.s,
        ExxDiv::None,
        &Default::default(),
    )
    .unwrap();
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let cfg = scfg(&fx.grid, None);
    let run = |f: &str, scf: &KScfResult| {
        kpoint_rks_stress(
            &fx.cell,
            &fx.prep,
            &fx.mesh,
            &hcfg,
            &fx.hk,
            KStressJk::Dense(&eri),
            scf,
            f,
            &cfg,
        )
    };
    for (name, key) in [("HSE06", "range-separated"), ("wB97X-V", "VV10")] {
        let e = run(name, &fx.res.scf)
            .err()
            .unwrap_or_else(|| panic!("{name} accepted"))
            .to_string();
        assert!(e.contains(key), "{name}: {e}");
    }
    // Guard, both sides: the SCAN result is accepted as SCAN and refused as
    // PBE / r2SCAN (the missing or different V^tau is seen).
    let ok = run("SCAN", &fx.res.scf).unwrap();
    assert!(ok.hf.fock_mismatch < 1e-12, "{:e}", ok.hf.fock_mismatch);
    for wrong in ["PBE", "R2SCAN"] {
        let e = run(wrong, &fx.res.scf).expect_err(wrong).to_string();
        assert!(e.contains("differs"), "{wrong}: {e}");
    }
    let mut t = fx.res.scf.clone();
    t.fock[1][(0, 1)] += Complex64::new(1e-6, 0.0);
    assert!(run("SCAN", &t).is_err(), "1e-6 tamper must be refused");
    // A Fock built without V^tau (defect inside the stress): skipped guard.
    let pulay = stress(&fx, Some(KXcStressMutation::TauNoVxc));
    assert!(pulay.hf.fock_mismatch == 0.0);
}

/// Production-wired mutants against the FD-anchored base.
#[test]
fn mutants_are_caught() {
    let fx = fixture(
        &H2,
        [1, 1, 3],
        "SCAN",
        Jk::Dense,
        atom_grid(20, 50, Some(CUTOFF)),
    );
    let base = stress(&fx, None);
    for m in [
        KXcStressMutation::NoTauStrain,
        KXcStressMutation::TauDoubled,
        KXcStressMutation::TauNoVxc,
        KXcStressMutation::NoWeightDeriv,
        KXcStressMutation::NoPointMotion,
        KXcStressMutation::WrongFactor,
        KXcStressMutation::NoHessian,
        KXcStressMutation::NoStrainMoment,
    ] {
        let d = mdiff(&stress(&fx, Some(m)).de_deps, &base.de_deps);
        eprintln!("{m:?}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "{m:?} survives: {d:e}");
    }
}

// ------------------------------------------------------------- p shells

/// Fixed-density XC stress FD with s+p shells on the TRICLINIC cell (uniform
/// grid, SCAN, all nine components). Physics hypothesis: residual at the FD
/// floor (measured below). Artifact hypothesis: an (a,b)-transposed tau strain
/// term is an identity when the moments are symmetric in (a,b) (cubic cell,
/// s shells) and misses here; the cubic s-only tests above do not see it.
/// The atom grid needs `neighbour_cutoff >= 14` for a smooth FD (cutoff 10
/// puts a 3.6e-5 mask jump in the energy), so this is a uniform-grid test.
#[test]
#[ignore = "16-function s+p SCAN SCF on the 4-atom triclinic cell, a fixed-D XC FD (18 strained builds) and three stresses; measured 217 s alone on a shared box, run serially on a quiet box"]
fn p_shell_triclinic_fixed_density_xc_stress_matches_fd() {
    let bs = sp_basis_h();
    let grid = KXcGrid::Uniform([24; 3]);
    let fx = fixture_bs(
        triclinic_cell(),
        [1, 1, 2],
        "SCAN",
        Jk::Dense,
        grid.clone(),
        ExxDiv::None,
        &bs,
    );
    let s = stress(&fx, None);
    let d0 = fx.res.scf.densities.clone();
    let fd = fd_xc_all9_bs(&fx.cell, &bs, [1, 1, 2], "SCAN", &grid, &d0, 1e-4);
    let miss = mdiff(&xc_part(&s), &fd);
    eprintln!(
        "triclinic s+p SCAN uniform 24^3: |xc - FD| {miss:.2e}; |xc| max {:.3e}",
        mmax(&xc_part(&s))
    );
    assert!(miss < P_SHELL_BAR, "{miss:e}");
    for m in [
        KXcStressMutation::TauTranspose,
        KXcStressMutation::NoTauStrain,
        KXcStressMutation::NoHessian,
    ] {
        let mm = mdiff(&xc_part(&stress(&fx, Some(m))), &fd);
        eprintln!("  triclinic s+p {m:?}: |xc - FD| {mm:.2e}");
        assert!(mm > MUTANT_BAR, "{m:?} survives: {mm:e}");
    }
}

#[test]
#[ignore = "second 61^3 PySCF-anchor row (r2SCAN); one SCF plus two stresses on a 61^3 grid, measured about the SCAN row's time"]
fn uniform_grid_stress_matches_pyscf_fftdf_krks_stress_r2scan() {
    pyscf_mgga_case("R2SCAN", "R2SCAN");
}

/// TPSS dE/d eps against FD of the converged TPSS SCF energy (uniform 24^3,
/// two components), and the tau defects against the same FD. Physics: the
/// FD floor (SCAN's on this fixture is in `meta_gga_stress_matches_fd_of_the_scf_energy`).
/// Artifact: a dropped / doubled tau strain term or V^tau missing from the
/// Pulay Fock misses by >= 1e-4. MEASURED (2026-10-22): correct 1.1e-8 (bar
/// 5e-7, shared with the SCAN row); misses `NoTauStrain` 5.0e-2, `TauDoubled`
/// 5.0e-2, `TauNoVxc` 2.3e-2, `NoPointMotion` 5.2e-1. `StressTauPath` (take
/// the tau path) is an EQUIVALENT mutant on a meta-GGA (the path is already
/// taken): measured 0, asserted.
#[test]
fn tpss_stress_matches_fd_of_the_scf_energy() {
    let grid = KXcGrid::Uniform([24, 24, 24]);
    let f = "TPSS";
    let fx = fixture(&H2, [1, 1, 3], f, Jk::Dense, grid.clone());
    let s = stress(&fx, None);
    let h = 1e-4;
    let comps = [(0, 0), (2, 2)];
    let fd: Vec<f64> = comps
        .iter()
        .map(|&(i, j)| {
            (energy_on(&strained(&fx.cell, i, j, h), [1, 1, 3], f, Jk::Dense, &grid)
                - energy_on(
                    &strained(&fx.cell, i, j, -h),
                    [1, 1, 3],
                    f,
                    Jk::Dense,
                    &grid,
                ))
                / (2.0 * h)
        })
        .collect();
    let miss = |t: &KRksStress| {
        comps.iter().zip(&fd).fold(0.0_f64, |m, (&(i, j), v)| {
            m.max((t.de_deps[i][j] - v).abs())
        })
    };
    let worst = miss(&s);
    eprintln!("TPSS uniform 24^3: |analytic - FD of SCF energy| {worst:.2e}");
    assert!(worst < 5e-7, "{worst:e}");
    for m in [
        KXcStressMutation::NoTauStrain,
        KXcStressMutation::TauDoubled,
        KXcStressMutation::TauNoVxc,
        KXcStressMutation::NoPointMotion,
    ] {
        let d = miss(&stress(&fx, Some(m)));
        eprintln!("  TPSS {m:?}: miss vs FD {d:.2e}");
        assert!(d > MUTANT_BAR, "{m:?} survives: {d:e}");
    }
    let d = mdiff(
        &stress(&fx, Some(KXcStressMutation::StressTauPath)).de_deps,
        &s.de_deps,
    );
    eprintln!("  TPSS StressTauPath (identity on a meta-GGA): {d:.2e}");
    assert!(d < 1e-12, "{d:e}");
}

// ------------------------------------------------------------- RS-GDF J

// The tau strain term lives in the grid XC response (`GridXc`), which never
// sees the J route; RS-GDF changes only the HF assembly (the fit's Coulomb
// pieces), anchored for PBE in `pbc_kxcstress.rs`. PHYSICS hypothesis: the
// SCAN RS-GDF stress is the derivative of the SCAN RS-GDF energy (FD at the
// dense-route floor), and the route difference (RS-GDF - dense) of the
// stress is the fit error, nearly the same for SCAN as for PBE on the same
// cell/aux. ARTIFACT hypotheses: a route-dependent tau defect (e.g. V^tau
// missing from the Pulay Fock on the fit route) misses FD by the size of the
// tau term and breaks the transfer by the same amount; a "pass" because the
// dense code ran is excluded by asserting the RS-GDF stress differs from the
// dense one by more than the transfer gap.

/// Total dE/d eps of a SCAN RS-GDF SCF against FD of its energy (uniform grid,
/// the dense test's step and bar), with the tau defects on the RS-GDF route
/// missing the same FD. Measured: (2,2) 1.1e-8, (0,2) 2.0e-10 (bar 5e-7);
/// all nine (ignored) worst 1.1e-8 SCAN and r2SCAN. Mutants against the same
/// FD: NoTauStrain 6.0e-2, TauDoubled 6.0e-2, TauNoVxc 5.6e-4.
#[test]
fn rsgdf_meta_gga_stress_matches_fd_of_the_scf_energy() {
    rsgdf_scf_fd_case("SCAN", &[(2, 2), (0, 2)], true);
}

/// All nine components, SCAN and r2SCAN, on RS-GDF.
#[test]
#[ignore = "SCAN + r2SCAN RS-GDF all-nine FD: 2 x 18 strained SCFs on a 24^3 uniform grid"]
fn rsgdf_meta_gga_stress_matches_fd_all_nine() {
    for f in ["SCAN", "R2SCAN"] {
        rsgdf_scf_fd_case(f, &ALL9, false);
    }
}

/// Fit-error transfer: `(sigma_RSGDF - sigma_dense)` for SCAN equals the same
/// difference for PBE on the same cell, aux and grid, up to the (small)
/// density dependence of the fit error. Bar: `TRANSFER_BAR` (measured below);
/// a tau defect on the RS-GDF route alone must break it. A tau defect common
/// to BOTH routes cancels in the difference: this test sees only
/// route-dependent defects, and the FD test above is the load-bearing anchor.
#[test]
fn rsgdf_minus_dense_stress_transfers_from_pbe_to_scan() {
    let grid = atom_grid(20, 50, Some(CUTOFF));
    let route_diff = |f: &str, m: Option<KXcStressMutation>| {
        let g = stress(&fixture(&H2, [1, 1, 3], f, Jk::Gdf, grid.clone()), m);
        let d = stress(&fixture(&H2, [1, 1, 3], f, Jk::Dense, grid.clone()), None);
        let mut o = g.de_deps;
        for i in 0..3 {
            for j in 0..3 {
                o[i][j] -= d.de_deps[i][j];
            }
        }
        o
    };
    let pbe = route_diff("PBE", None);
    let scan = route_diff("SCAN", None);
    let gap = mdiff(&scan, &pbe);
    eprintln!(
        "RS-GDF - dense: |PBE| {:.3e}, |SCAN| {:.3e}, |SCAN - PBE| {gap:.2e}",
        mmax(&pbe),
        mmax(&scan)
    );
    // The route difference is resolved (the RS-GDF route really ran) and the
    // transfer holds far below it.
    assert!(mmax(&pbe) > 10.0 * TRANSFER_BAR, "{:e}", mmax(&pbe));
    assert!(gap < TRANSFER_BAR, "{gap:e}");
    for m in [KXcStressMutation::NoTauStrain, KXcStressMutation::TauNoVxc] {
        let mm = mdiff(&route_diff("SCAN", Some(m)), &pbe);
        eprintln!("  RS-GDF-only {m:?}: |SCAN - PBE| {mm:.2e}");
        assert!(mm > MUTANT_BAR, "{m:?} survives the transfer: {mm:e}");
    }
}
