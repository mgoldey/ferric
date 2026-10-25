//! Restricted open-shell range-separated hybrids (ROKS), ENERGY only, at
//! Gamma (`gamma_roks_rsh`) and on a k-mesh (`solve_kroks` with HSE06 /
//! CAM-B3LYP), mgoldey/ferric#354. The always-on subset; the weekly rest
//! (remaining PySCF pins, supercell, RS-GDF, more mutants) is
//! `validation_roks_rsh.rs`. Cells, grids and exchange setup are those of the
//! UKS tests (`oshell_rsh_support`), so ROKS and UKS of a row are the same
//! physical problem.
//!
//! # Hypotheses (written before any number was measured)
//!
//! ROKS builds the SAME per-spin Focks as UKS, `F_σ = h + J[D] − K[D_σ] +
//! V_σ` with the attenuated `K[D_σ]` scaled by 1 (fractions and the MIXED
//! Madelung constant inside), and only then couples them in the Roothaan
//! effective Fock. So nothing new is computed in the exchange; what is new is
//! that the ROKS SCF must route the attenuated K (instead of `a·K_Coulomb`)
//! into both spin Focks and use the mixed Madelung constant for the staged
//! start and the gap report.
//!
//! * Physics: a singlet ROKS equals RKS to SCF precision; the 1x1x1 mesh
//!   equals Gamma ROKS; an N-point mesh equals Gamma ROKS of the supercell /
//!   N; PySCF `pbc.dft.ROKS` / `KROKS` agree; and the variational ordering
//!   E(ROKS) >= E(UKS) holds on the same cell (a sanity check, not a bar).
//! * Artifact: the K routed per spin wrongly (alpha K on the beta Fock, beta
//!   K dropped), the plain Coulomb Madelung constant instead of the mixed one,
//!   the fractions applied twice (attenuated K scaled by c_sr), or the bare
//!   alpha Fock instead of the Roothaan one: the first and the last are
//!   invisible in the closed-shell limit, so all are measured on the
//!   POLARIZED H3 doublet against the PySCF pin.
//!
//! # What each anchor does and does not test
//!
//! * The exchange kernel is the closed-shell one (shared with RKS and UKS):
//!   only PySCF is independent of it. The closed-shell, 1x1x1 and supercell
//!   anchors test the ROKS routing, not the kernel.
//! * As in the UKS file: the erf kernel at G != 0 is live only for CAM-B3LYP
//!   on this cell (HSE06's largest erf weight is 4e-16 at Gamma), so the
//!   always-on pins and mutants use CAM-B3LYP; HSE06 rows check the mixed
//!   Madelung constant and the assembly.
//!
//! # Measured (2026-10-18, PySCF 2.13.1 pins read from the JSON)
//!
//! E_ferric - E_PySCF: <= 4.4e-13 over all 11 ROKS rows (Gamma H3 HSE06
//! 3.4e-13 / CAM-B3LYP 3.3e-13, exxdiv none 3.7e-13; H4 triplet 4.4e-13 /
//! 4.3e-13; 1x1x2 1.8e-13 .. 2.4e-13). PySCF converged every row, KROKS
//! included. ROKS - RKS (closed shell): <= 4.0e-15 (H2 Gamma / 1x1x2, both
//! exxdiv; H4 singlet -8.9e-16). Gamma - 1x1x1: 0.0 (E_xc 5.6e-16). 1x1x2 -
//! supercell/2: 1.7e-14 (HSE06), 7.1e-15 (CAM). ROKS - UKS (PySCF pins on
//! the same row): +5.1e-6 .. +3.1e-3, positive on all 9 rows with both pins.
//! RS-GDF - dense, ROKS RSH over the PBE0 UKS plain-Coulomb fit error on the
//! same cell: 0.92 / 1.84 (HSE06 / CAM, 1x1x2), 0.99 / 1.11 (Gamma).
//!
//! # Mutation ledger (E - PySCF pin, polarized CAM-B3LYP H3 doublet)
//!
//! Test-only switches (Gamma `GammaSpinMutant`, k `KRohfMutation` at 1x1x1):
//! alpha K on the beta Fock -1.16e-2 (both paths), beta K dropped +1.49e-1
//! (both), Madelung halved per spin (Gamma) +2.15e-1, plain instead of mixed
//! Madelung (k) -5.03e-1, attenuated K scaled by c_sr (fractions twice, k)
//! +3.93e-1, bare alpha Fock instead of Roothaan (k) +7.4e-4. Alpha-on-beta
//! and bare-alpha are 0.0 in the closed-shell limit (H2 1x1x2, validation
//! file). Source mutants (applied by hand, then reverted): Gamma
//! `run_staged` using the plain Madelung constant -> HSE06 pin -7.5e-1 and
//! Gamma - 1x1x1 -5.0e-1; Gamma ROKS not replacing K by the attenuated one
//! (full Coulomb K, a = 1) -> HSE06 pin -2.1e-1; k-point `drive` not setting
//! `dense.rsh` -> CAM 1x1x2 pin -6.6e-1 and closed-shell KROKS - KRKS
//! (`pbc_krsh.rs`) -5.6e-1. Bars: pins and closed shell 1e-11 (23x the worst
//! miss), 1x1x1 1e-11, supercell 1e-10; a mutant must move E by > 1e-4 (the
//! smallest, bare alpha, is 7.4e-4).
//!
//! Runs in every shard (~55 s): H4 singlet closed-shell limit (Gamma and
//! 1x1x2), Gamma H3 HSE06 / CAM pins with E(ROKS) >= E(UKS), the CAM 1x1x2
//! KROKS pin, 1x1x1 == Gamma, the mutants and the by-name refusals.

mod common;
#[path = "oshell_rsh_support/mod.rs"]
mod support;

use common::*;
use support::*;

use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::gamma_rsh::GammaSpinMutant;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kroks::KRohfMutation;

// ------------------------------------------------ closed-shell limit

/// H4 singlet in the triclinic cell (4 AOs, 2 occupied orbitals not fixed by
/// symmetry): ROKS must be RKS at Gamma and on a 1x1x2 mesh.
#[test]
fn closed_h4_singlet_roks_is_rks_at_gamma_and_on_a_mesh() {
    let c = cell_h4_singlet();
    let o = gamma_roks(&c, "CAM_B3LYP", ExxDiv::Ewald, GRun::default());
    let r = gamma_rks_energy(&c, "CAM_B3LYP", ExxDiv::Ewald, GRun::default());
    eprintln!("H4 singlet Gamma CAM: RO-R {:.3e}", o.scf.energy - r);
    assert!((o.scf.energy - r).abs() < CLOSED_TOL, "Gamma");
    let n = [1, 1, 2];
    let ok = kroks_run(&c, "CAM_B3LYP", n, ExxDiv::Ewald, false, None)
        .scf
        .energy;
    let rk = krks_energy(&c, "CAM_B3LYP", n, ExxDiv::Ewald, false);
    eprintln!("H4 singlet 1x1x2 CAM: RO-R {:.3e}", ok - rk);
    assert!((ok - rk).abs() < CLOSED_TOL, "1x1x2");
}

// ------------------------------------------------ polarized pins

#[test]
fn gamma_roks_matches_pyscf_h3_doublet() {
    let cell = cell_named("h3d");
    for f in ["HSE06", "CAM_B3LYP"] {
        let r = gamma_roks(&cell, f, ExxDiv::Ewald, GRun::default());
        let want = roks_pin("h3d", f, [1, 1, 1], ExxDiv::Ewald).expect("PySCF converged");
        let d = r.scf.energy - want;
        eprintln!(
            "H3 doublet {f} Gamma ROKS: dE vs PySCF {d:.2e}, <S2> {:.6}",
            r.s2
        );
        assert!(d.abs() < PIN_TOL, "{f}: {d:.2e}");
        assert!((r.s2 - 0.75).abs() < 1e-8, "ROKS is spin-pure");
        // Variational ordering: UKS relaxes the ROKS constraint.
        let u = pin("h3d", f, [1, 1, 1], ExxDiv::Ewald);
        assert!(
            r.scf.energy >= u,
            "{f}: E_ROKS {} < E_UKS {u}",
            r.scf.energy
        );
    }
}

#[test]
fn kpoint_roks_cam_b3lyp_1x1x2_doublet_matches_pyscf() {
    let cell = cell_named("h3d");
    let n = [1, 1, 2];
    let r = kroks_run(&cell, "CAM_B3LYP", n, ExxDiv::Ewald, false, None);
    let want = roks_pin("h3d", "CAM_B3LYP", n, ExxDiv::Ewald).expect("PySCF converged");
    let d = r.scf.energy - want;
    eprintln!(
        "H3 doublet CAM 1x1x2 KROKS: dE vs PySCF {d:.2e} ({} it)",
        r.scf.iterations
    );
    assert!(d.abs() < PIN_TOL, "{d:.2e}");
}

#[test]
fn one_point_mesh_is_gamma_roks_rsh() {
    let cell = cell_named("h3d");
    let g = gamma_roks(&cell, "CAM_B3LYP", ExxDiv::Ewald, GRun::default());
    let k = kroks_run(&cell, "CAM_B3LYP", [1, 1, 1], ExxDiv::Ewald, false, None);
    let d = g.scf.energy - k.scf.energy;
    eprintln!(
        "H3 doublet CAM ROKS: Gamma - 1x1x1 {d:.2e}; E_xc {:.2e}",
        g.e_xc - k.e_xc
    );
    assert!(d.abs() < ONE_POINT_TOL);
    assert!((g.e_xc - k.e_xc).abs() < ONE_POINT_EXC_TOL);
    assert!((g.madelung - k.madelung).abs() < 1e-10, "mixed v_M");
    let c_sr = rsh_of("CAM_B3LYP").c_sr;
    assert!((k.exact_exchange_fraction - c_sr).abs() < 1e-15, "k: c_sr");
    assert!(
        (g.exact_exchange_fraction - c_sr).abs() < 1e-15,
        "Gamma: c_sr"
    );
}

// ------------------------------------------------ mutants

/// A mutant must move the polarized CAM-B3LYP energy away from the PySCF pin
/// by more than this: 1e7 x the pin bar, 7x below the smallest measured shift
/// (bare alpha Fock, 7.4e-4).
const ROKS_MUT_MIN: f64 = 1e-4;

/// Each is the mutant's polarized CAM-B3LYP energy minus the PySCF pin; a
/// mutant must move E by more than `ROKS_MUT_MIN`.
#[test]
fn roks_mutants_fail_the_polarized_cam_pin() {
    let cell = cell_named("h3d");
    let want = roks_pin("h3d", "CAM_B3LYP", [1, 1, 1], ExxDiv::Ewald).unwrap();
    for m in [
        GammaSpinMutant::AlphaOnBeta,
        GammaSpinMutant::DropBeta,
        GammaSpinMutant::MadelungHalf,
    ] {
        let run = GRun {
            spin_mutant: Some(m),
            ..Default::default()
        };
        let d = gamma_roks(&cell, "CAM_B3LYP", ExxDiv::Ewald, run)
            .scf
            .energy
            - want;
        eprintln!("Gamma ROKS mutant {m:?}: E - PySCF pin {d:+.3e}");
        assert!(d.abs() > ROKS_MUT_MIN, "{m:?} not detected (shift {d:.2e})");
    }
    for m in [
        KRohfMutation::KAlphaOnBeta,
        KRohfMutation::KDropBeta,
        KRohfMutation::PlainMadelungForRsh,
        KRohfMutation::RshScaledBySr,
        KRohfMutation::BareAlphaFock,
    ] {
        let d = kroks_run(&cell, "CAM_B3LYP", [1, 1, 1], ExxDiv::Ewald, false, Some(m))
            .scf
            .energy
            - want;
        eprintln!("1x1x1 KROKS mutant {m:?}: E - PySCF pin {d:+.3e}");
        assert!(d.abs() > ROKS_MUT_MIN, "{m:?} not detected (shift {d:.2e})");
    }
}

// ------------------------------------------------ refusals

/// The plain global-hybrid ROKS entry (`gamma_roks`) still refuses a
/// range-separated name (it would otherwise run it with no attenuated K), and
/// so do the plain Gamma ROKS force and stress on a REAL ROKS RSH SCF. The
/// k-point ROKS FORCE and STRESS cover RSH on dense-AFT J/K
/// (`pbc_kroks_rsh_force.rs`, `pbc_kuks_rsh_stress.rs`; at Gamma
/// `gamma_rsh::gamma_roks_rsh_gradient`): without the SCF's exxdiv each is a
/// typed refusal, with it and the SCF's own grid each differentiates this
/// state (Fock guard and orbital-gradient gate passed).
#[test]
fn plain_gamma_roks_and_roks_rsh_derivatives_refuse_by_name() {
    use ferric_pbc::kgrad::KGradJk;
    use ferric_pbc::kroks::{solve_kroks_on_grid_keep, KRoksConfig};
    use ferric_pbc::kscf::KRetainedJk;
    use ferric_pbc::kstress::KStressJk;
    use ferric_pbc::kuxcgrad::{kpoint_roks_gradient, KUksGradConfig};
    use ferric_pbc::kuxcstress::{kpoint_roks_stress, KUksStressConfig};
    let named = |m: String| assert!(m.contains("range-separated"), "{m}");
    let cell = cell_named("h3d");
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA_HCORE);

    let s = GSetup::new(&cell, "CAM_B3LYP", ExxDiv::Ewald, GRun::default());
    let cfg = ferric_pbc::rohf::GammaRoksConfig::new("HYB_GGA_XC_CAM_B3LYP");
    named(
        ferric_pbc::rohf::gamma_roks(&cell, &s.prep, &s.hc, s.ints(), &cfg)
            .unwrap_err()
            .to_string(),
    );
    let r = gamma_roks(&cell, "CAM_B3LYP", ExxDiv::Ewald, GRun::default());
    let eri = s.dense.as_ref().unwrap();
    named(
        ferric_pbc::grad::gamma_roks_gradient(&cell, &s.prep, &hcfg, &s.hc, eri, &r.scf, &cfg)
            .unwrap_err()
            .to_string(),
    );
    named(
        ferric_pbc::stress::gamma_roks_stress(
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

    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 1]).unwrap();
    let mut kcfg = KRoksConfig::new(&cell, ExxDiv::Ewald, "HYB_GGA_XC_CAM_B3LYP");
    kcfg.rohf.hcore = hcfg;
    let (kr, kept) = solve_kroks_on_grid_keep(&cell, &prep, None, &mesh, &s.grid, &kcfg).unwrap();
    let KRetainedJk::Dense(e) = &kept.jk else {
        panic!("dense J/K expected")
    };
    let grad = KUksGradConfig {
        base: Default::default(),
        spin_mutation: None,
    };
    let call = |g: &KUksGradConfig| {
        kpoint_roks_gradient(
            &cell,
            &prep,
            &mesh,
            &hcfg,
            &kept.hk,
            KGradJk::Dense(e),
            &kr.scf,
            "HYB_GGA_XC_CAM_B3LYP",
            g,
        )
    };
    let ge = call(&grad).unwrap_err().to_string();
    assert!(ge.contains("needs KRksGradConfig::exxdiv"), "{ge}");
    let mut ok = grad.clone();
    ok.base.exxdiv = Some(ExxDiv::Ewald);
    ok.base.grid = ferric_pbc::kxcgrad::KXcGrid::Uniform([MESH; 3]);
    let g = call(&ok).unwrap_or_else(|e| panic!("ROKS RSH force on the SCF's own state: {e}"));
    assert!(g.grad.iter().all(|v| v.is_finite()) && g.hf.fock_mismatch < 1e-8);
    // The k-point ROKS STRESS covers range-separated hybrids too (#492,
    // `pbc_kuks_rsh_stress.rs`): without the SCF's exxdiv a typed refusal,
    // with it and the SCF's own grid it differentiates this state.
    let scall = |c: &KUksStressConfig| {
        kpoint_roks_stress(
            &cell,
            &prep,
            &mesh,
            &hcfg,
            &kept.hk,
            KStressJk::Dense(e),
            &kr.scf,
            "HYB_GGA_XC_CAM_B3LYP",
            c,
        )
    };
    let se = scall(&KUksStressConfig::default()).unwrap_err().to_string();
    assert!(se.contains("needs KRksStressConfig::exxdiv"), "{se}");
    let mut sok = KUksStressConfig::default();
    sok.base.exxdiv = Some(ExxDiv::Ewald);
    sok.base.grid = ferric_pbc::kxcgrad::KXcGrid::Uniform([MESH; 3]);
    let st = scall(&sok).unwrap_or_else(|e| panic!("ROKS RSH stress on the SCF's own state: {e}"));
    assert!(st.de_deps.iter().flatten().all(|v| v.is_finite()) && st.hf.fock_mismatch < 1e-8);
}
