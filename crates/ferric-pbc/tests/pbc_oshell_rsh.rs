//! Open-shell range-separated hybrids (UKS), ENERGY only, at Gamma
//! (`gamma_uks_rsh`) and on a k-mesh (`solve_kuks` with HSE06 / CAM-B3LYP),
//! mgoldey/ferric#354. The always-on subset; the weekly rest (remaining PySCF
//! pins, k-point mutants, supercell, RS-GDF) is `validation_oshell_rsh.rs`.
//!
//! # Hypotheses (written before any number was measured)
//!
//! The attenuated K builder is linear in the density and carries the
//! fractions and the MIXED Madelung constant, so the open-shell step is
//! `K_sigma = K[D_sigma]`, scaled by 1.
//!
//! * Physics: a singlet UKS equals RKS to SCF precision; the 1x1x1 mesh
//!   equals the Gamma UKS; an N-point mesh equals the Gamma UKS of the
//!   supercell / N; PySCF's polarized pins agree.
//! * Artifact: a per-spin error (alpha K on the beta Fock, a dropped beta
//!   term, a halved per-spin Madelung constant, a swapped spin build, the
//!   plain instead of the mixed Madelung constant) can leave the closed-shell
//!   anchors intact (D_alpha = D_beta) and move only polarized energies. So
//!   the mutants are measured on POLARIZED cells against PySCF pins.
//!
//! # What each anchor does and does not test
//!
//! * H2 closed-shell UKS == RKS: H2 is inversion symmetric with 2 AOs, so the
//!   occupied orbital is fixed by symmetry. It tests the energy ASSEMBLY of the
//!   per-spin path (K scaling, Madelung, XC), not SCF freedom; it is bit-
//!   identical (0.0) on its four rows. The H4 singlet (4 AOs, 2 occupied, no
//!   symmetry fix) at Gamma and 1x1x2 does have SCF freedom (-8.9e-16 and
//!   -4.0e-15). Of the spin mutants only alpha-on-beta and the spin swap are
//!   invisible in the closed-shell limit (2.2e-16, validation file);
//!   beta-dropped and Madelung-halved move it too.
//! * The attenuated kernel at G != 0 is exercised ONLY by CAM-B3LYP
//!   (omega 0.33). For HSE06 (omega 0.11) on this cell the largest erf weight
//!   exp(-|q+G|^2/4w^2) is 4.2e-16 at Gamma (1.4e-4 at 1x1x2) against 2.0e-2 /
//!   0.37 for CAM-B3LYP, so for HSE06 the erf piece enters through the mixed
//!   Madelung constant v_M(omega) and the omega seen by the semilocal part,
//!   not through the G != 0 kernel. HSE06-minus-PBE0 (-2.4e-5 on H3, 7.1e-8
//!   on H4 at Gamma) is therefore mostly the screened semilocal exchange. Every
//!   open-shell test here that checks the G != 0 kernel or the omega / fraction
//!   mutants uses CAM-B3LYP; the HSE06 rows check the mixed Madelung constant
//!   and the per-spin assembly. No open-shell wB97X run exists.
//! * Both the Gamma and the k-point open-shell paths call the SAME attenuated
//!   K kernel as the closed-shell code, so only PySCF is independent of it.
//!
//! # Measured (2026-10-18, release, 6-core box, PySCF 2.13.1 pins read from the
//! # JSON at full precision)
//!
//! E_ferric - E_PySCF: <= 4.4e-13 over all 16 energies (Gamma H3 HSE06 3.3e-13,
//! CAM-B3LYP 3.3e-13; H3 CAM 1x1x2 1.7e-13; H4 triplet 4.3e-13 / 4.4e-13; PBE0
//! 3.7e-13). Gamma - 1x1x1: 3.3e-15 (CAM, H3). Mutant shifts E - pin on the
//! polarized CAM-B3LYP H3 cell at Gamma: alpha K on beta -1.08e-2, beta K
//! dropped +1.49e-1, Madelung/2 per spin +2.15e-1 (always-on); omega x1.01
//! -2.3e-3, omega x0.01 +2.5e-1, c_lr := c_sr +2.5e-1, c_sr x0.99 +1.3e-3
//! (validation). Bars: pins 1e-11 (23x the worst miss, 5e6 below the finest
//! mutant); spin mutants must move E by > 1e-3 (smallest 1.1e-2); closed-shell
//! 1e-11. No spin swap exists at Gamma (the SCF calls K once per spin, alpha
//! then beta); the k-point code has it.
//!
//! Runs in every shard (~45 s): closed-shell H2 and H4 limits, Gamma pins
//! (H3 HSE06 and CAM-B3LYP), the CAM-B3LYP 1x1x2 pin, the Gamma spin mutants,
//! 1x1x1 == Gamma, and the by-name refusals (gradient and stress of a real
//! open-shell RSH SCF, Gamma and k). ROKS with these functionals is
//! `pbc_roks_rsh.rs`.

mod common;
#[path = "oshell_rsh_support/mod.rs"]
mod support;

use common::*;
use support::*;

use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::gamma_rsh::GammaSpinMutant;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kpts::KPointMesh;

// ------------------------------------------------ closed-shell limit

/// H2 (inversion-symmetric, 2 AOs): the occupied orbital is fixed by
/// symmetry, so this checks the ENERGY ASSEMBLY of the per-spin path (K
/// scaling, Madelung, XC), not SCF freedom; `closed_h4_*` below has SCF
/// freedom. Measured: bit-identical (0.0) on all four (functional, exxdiv)
/// rows of this one cell.
#[test]
fn gamma_closed_shell_h2_uks_is_rks() {
    closed_limit_case("HSE06", ExxDiv::Ewald);
    closed_limit_case("CAM_B3LYP", ExxDiv::None);
}

/// H4 singlet in the triclinic cell: 4 AOs, 2 occupied orbitals not fixed by
/// symmetry, so UKS and RKS can differ if the per-spin path is wrong.
#[test]
fn closed_h4_singlet_uks_is_rks_at_gamma_and_on_a_mesh() {
    let c = cell_h4_singlet();
    let u = gamma_uks(&c, "CAM_B3LYP", ExxDiv::Ewald, GRun::default());
    let r = gamma_rks_energy(&c, "CAM_B3LYP", ExxDiv::Ewald, GRun::default());
    eprintln!(
        "H4 singlet Gamma CAM: U-R {:.3e} (<S2> {:.1e})",
        u.scf.energy - r,
        u.s2
    );
    assert!((u.scf.energy - r).abs() < CLOSED_TOL, "Gamma");
    let n = [1, 1, 2];
    let uk = kuks_run(&c, "CAM_B3LYP", n, ExxDiv::Ewald, false, None)
        .scf
        .energy;
    let rk = krks_energy(&c, "CAM_B3LYP", n, ExxDiv::Ewald, false);
    eprintln!("H4 singlet 1x1x2 CAM: U-R {:.3e}", uk - rk);
    assert!((uk - rk).abs() < CLOSED_TOL, "1x1x2");
}

// ------------------------------------------------ polarized pins

#[test]
fn gamma_polarized_matches_pyscf_h3_doublet() {
    let cell = cell_named("h3d");
    for f in ["HSE06", "CAM_B3LYP"] {
        let r = gamma_uks(&cell, f, ExxDiv::Ewald, GRun::default());
        let d = r.scf.energy - pin("h3d", f, [1, 1, 1], ExxDiv::Ewald);
        eprintln!(
            "H3 doublet {f} Gamma: dE vs PySCF {d:.2e}, <S2> {:.4}",
            r.s2
        );
        assert!(d.abs() < PIN_TOL, "{f}: {d:.2e}");
        assert!(
            r.s2 > 0.75 && r.s2 < 0.76,
            "doublet, not a closed-shell mix"
        );
    }
}

/// Multi-k per-spin assembly, the erf kernel at q != 0 and the mixed
/// Madelung constant, against PySCF (CAM-B3LYP, whose erf weight is live).
#[test]
fn kpoint_cam_b3lyp_1x1x2_doublet_matches_pyscf() {
    let cell = cell_named("h3d");
    let n = [1, 1, 2];
    let r = kuks_run(&cell, "CAM_B3LYP", n, ExxDiv::Ewald, false, None);
    let d = r.scf.energy - pin("h3d", "CAM_B3LYP", n, ExxDiv::Ewald);
    eprintln!("H3 doublet CAM 1x1x2: dE vs PySCF {d:.2e}");
    assert!(d.abs() < PIN_TOL, "{d:.2e}");
}

/// Each is the unmutated polarized CAM-B3LYP energy minus the mutant's; a
/// mutant must move E by more than `MUT_MIN`.
#[test]
fn gamma_spin_mutants_fail_the_polarized_cam_pin() {
    let cell = cell_named("h3d");
    let want = pin("h3d", "CAM_B3LYP", [1, 1, 1], ExxDiv::Ewald);
    for m in [
        GammaSpinMutant::AlphaOnBeta,
        GammaSpinMutant::DropBeta,
        GammaSpinMutant::MadelungHalf,
    ] {
        let run = GRun {
            spin_mutant: Some(m),
            ..Default::default()
        };
        let d = gamma_uks(&cell, "CAM_B3LYP", ExxDiv::Ewald, run).scf.energy - want;
        eprintln!("Gamma mutant {m:?}: E - PySCF pin {d:+.3e}");
        assert!(d.abs() > MUT_MIN, "{m:?} not detected (shift {d:.2e})");
    }
}

#[test]
fn one_point_mesh_is_gamma_uks_rsh() {
    let cell = cell_named("h3d");
    let g = gamma_uks(&cell, "CAM_B3LYP", ExxDiv::Ewald, GRun::default());
    let k = kuks_run(&cell, "CAM_B3LYP", [1, 1, 1], ExxDiv::Ewald, false, None);
    let d = g.scf.energy - k.scf.energy;
    eprintln!(
        "H3 doublet CAM: Gamma - 1x1x1 {d:.2e}; E_xc {:.2e}",
        g.e_xc - k.e_xc
    );
    assert!(d.abs() < ONE_POINT_TOL);
    assert!((g.e_xc - k.e_xc).abs() < ONE_POINT_EXC_TOL);
    assert!((g.madelung - k.madelung).abs() < 1e-10, "mixed v_M");
    assert!(
        (k.exact_exchange_fraction - rsh_of("CAM_B3LYP").c_sr).abs() < 1e-15,
        "reported fraction is c_sr"
    );
}

// ------------------------------------------------ refusals

/// The derivatives of a REAL open-shell RSH SCF refuse by name (and so does
/// the plain global-hybrid ROKS entry; ROKS RSH is `gamma_roks_rsh`), except
/// the k-point UKS force, which covers RSH on dense-AFT J/K
/// (`pbc_kuks_rsh_force.rs`) and refuses here only for the missing exxdiv.
#[test]
fn open_shell_rsh_derivatives_refuse_by_name() {
    use ferric_pbc::kdense_aft::KDenseAftEri;
    use ferric_pbc::kgrad::KGradJk;
    use ferric_pbc::kscf::KRetainedJk;
    use ferric_pbc::kstress::KStressJk;
    use ferric_pbc::kuks::solve_kuks_on_grid_keep;
    use ferric_pbc::kuxcgrad::{kpoint_uks_gradient, KUksGradConfig};
    use ferric_pbc::kuxcstress::{kpoint_uks_stress, KUksStressConfig};
    let named = |m: String| assert!(m.contains("range-separated"), "{m}");
    let e = ferric_pbc::kxcgrad::check_kpoint_deriv_functional(true, "HSE06", "with_gradient")
        .unwrap_err();
    named(e);
    let cell = cell_named("h3d");
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA_HCORE);

    // Gamma: converge an open-shell CAM-B3LYP SCF, hand it to the derivatives.
    let s = GSetup::new(&cell, "CAM_B3LYP", ExxDiv::Ewald, GRun::default());
    let mut cfg = ferric_pbc::dft::GammaUksConfig::new("HYB_GGA_XC_CAM_B3LYP");
    cfg.scf.density_conv = 1e-9;
    let r = ferric_pbc::dft::gamma_uks_rsh_on_grid(
        &cell,
        &s.prep,
        &s.hc,
        s.ints(),
        &s.ex,
        &s.grid,
        &cfg,
    )
    .unwrap();
    let eri = s.dense.as_ref().unwrap();
    named(
        ferric_pbc::grad::gamma_uks_gradient(&cell, &s.prep, &hcfg, &s.hc, eri, &r.scf, &cfg)
            .unwrap_err()
            .to_string(),
    );
    named(
        ferric_pbc::stress::gamma_uks_stress(
            &cell,
            &s.prep,
            &hcfg,
            &s.hc,
            eri,
            &r.scf,
            &cfg,
            &ferric_pbc::stress::GammaStressConfig::default(),
        )
        .unwrap_err()
        .to_string(),
    );
    // The plain global-hybrid ROKS entry resolves the functional with the
    // global-hybrid gate before any grid or integral is built (RSH ROKS is
    // gamma_roks_rsh, pbc_roks_rsh.rs).
    let rcfg = ferric_pbc::rohf::GammaRoksConfig::new("HSE06");
    named(
        ferric_pbc::rohf::gamma_roks(&cell, &s.prep, &s.hc, s.ints(), &rcfg)
            .unwrap_err()
            .to_string(),
    );

    // k-point: a real KUKS CAM-B3LYP SCF handed to the force and the stress.
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 1]).unwrap();
    let mut kcfg = ferric_pbc::kuks::KUksConfig::new(&cell, ExxDiv::Ewald, "HYB_GGA_XC_CAM_B3LYP");
    kcfg.kuhf.hcore = hcfg;
    let (kr, kept) = solve_kuks_on_grid_keep(&cell, &prep, None, &mesh, &s.grid, &kcfg).unwrap();
    let KRetainedJk::Dense(e): &KRetainedJk = &kept.jk else {
        panic!("dense J/K expected")
    };
    let e: &KDenseAftEri = e;
    let grad = KUksGradConfig {
        base: Default::default(),
        spin_mutation: None,
    };
    // The k-point UKS FORCE covers range-separated hybrids on dense-AFT J/K
    // (`pbc_kuks_rsh_force.rs`); without the SCF's exxdiv it is still a
    // typed refusal, never a silent force.
    let ge = kpoint_uks_gradient(
        &cell,
        &prep,
        &mesh,
        &hcfg,
        &kept.hk,
        KGradJk::Dense(e),
        &kr.scf,
        "HYB_GGA_XC_CAM_B3LYP",
        &grad,
    )
    .unwrap_err()
    .to_string();
    assert!(ge.contains("needs KRksGradConfig::exxdiv"), "{ge}");
    named(
        kpoint_uks_stress(
            &cell,
            &prep,
            &mesh,
            &hcfg,
            &kept.hk,
            KStressJk::Dense(e),
            &kr.scf,
            "HYB_GGA_XC_CAM_B3LYP",
            &KUksStressConfig::default(),
        )
        .unwrap_err()
        .to_string(),
    );
}
