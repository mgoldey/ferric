//! Analytic RHF/UHF/ROHF Hessian with an effective core potential AND
//! density-fitted J/K (issue #406). Per-commit tier; the slow routes and the
//! PySCF comparisons are in `validation_ecp_df_hessian.rs`.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test ecp_df_hessian
//! ```
//!
//! # Why this composes
//!
//! `V_ECP` is a one-electron operator of the core Hamiltonian. The fitted J/K
//! Hessian replaces only the four-index skeleton and the `J^x`/`K^x` and CPHF
//! pieces; the ECP skeleton `Σ D ∂²V_ECP` and `∂V_ECP/∂x` in `F^x` are added by
//! the same shared code (`extra_skeleton`, `add_extra_first_order`) on the exact
//! and the fitted routes. ROHF additionally rebuilds its trusted spin Focks from
//! `h`, which must carry `V_ECP` (it did not).
//!
//! # Anchors
//!
//! * FD of ferric's own analytic fitted gradient (ECP in hcore on both sides),
//!   with the ECP-skeleton and `dV_ECP/dx` negative controls. A broken ECP term
//!   misses by 1e-2..1e0; the floor is 1e-7 (bars derived in `ecp_df_common`).
//! * The fitted ECP gradient against a central FD of ferric's fitted ENERGY (no
//!   shared gradient code).
//! * The probe: RI + ECP is analytic where the RI-K metric is conditioned, and
//!   `Auto` falls back to FD where it is not (HI / def2-universal-jkfit).
//!
//! Artifact hypothesis, written before measuring: a broken composition (ECP
//! term lost or doubled in the fitted route) misses FD by 1e-2..1e-1, a correct
//! one sits at the FD floor <1e-6.

mod ecp_df_common;

use ecp_df_common::*;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::{
    harmonic_frequencies, FrequencyConfig, FrequencyReference, HessianMethod, HessianSource,
};
use ferric_scf::hessian::{analytic_rohf_hessian_available, analytic_uhf_hessian_available};
use ferric_scf::rhf::RhfConfig;

#[test]
fn rhf_ri_jk_ecp_matches_fd_of_the_ri_gradient() {
    check_against_fd(
        "HOI RHF RI-JK",
        HOI,
        1,
        Reference::Rhf,
        Fit::Jk(AUX_OK),
        &[0, 4, 8],
    );
}

#[test]
fn uhf_ri_jk_ecp_matches_fd_of_the_ri_gradient() {
    check_against_fd(
        "HIH UHF RI-JK",
        HIH,
        2,
        Reference::Uhf,
        Fit::Jk(AUX_OK),
        &[0, 5, 8],
    );
}

/// ROHF exact J/K: the stored-vs-rebuilt Fock check failed by 25 Ha before
/// `trusted_focks` used `hcore_ecp`.
#[test]
fn rohf_exact_ecp_matches_fd_of_the_rohf_gradient() {
    check_against_fd(
        "HIH ROHF exact",
        HIH,
        2,
        Reference::Rohf,
        Fit::Exact,
        &[0, 5, 8],
    );
}

#[test]
fn rohf_ri_j_ecp_matches_fd_of_the_rohf_gradient() {
    check_against_fd(
        "HIH ROHF RI-J",
        HIH,
        2,
        Reference::Rohf,
        Fit::J(AUX_OK),
        &[0, 5, 8],
    );
}

/// RI-J-only is not the RI-JK Hessian (guards a `Fit::J` that silently fits K).
#[test]
fn ri_j_only_differs_from_ri_jk() {
    if !has_deriv2() {
        return;
    }
    let bs = basis::bundled("def2-svp").unwrap();
    let mol = ecp_mol(HOI, 1, &bs);
    let (fj, fjk) = (Fit::J(AUX_OK), Fit::Jk(AUX_OK));
    let hj = hessian(&solve(&mol, &bs, Reference::Rhf, fj), Reference::Rhf, fj);
    let hjk = hessian(&solve(&mol, &bs, Reference::Rhf, fjk), Reference::Rhf, fjk);
    let d = max_diff(&hj, &hjk);
    eprintln!("HOI RI-J vs RI-JK Hessian: {d:.3e}");
    assert!(d > 1e-5, "RI-J and RI-JK Hessians coincide ({d:.3e})");
}

/// The fitted ECP gradient against a 4th-order central FD of the fitted ENERGY.
/// Independent of every gradient and Hessian formula.
#[test]
fn ri_ecp_gradient_matches_fd_of_the_ri_energy() {
    let bs = basis::bundled("def2-svp").unwrap();
    let mol = ecp_mol(HOI, 1, &bs);
    let fit = Fit::Jk(AUX_OK);
    let s = solve(&mol, &bs, Reference::Rhf, fit);
    let g = gradient_of(&s, Reference::Rhf);
    let h = 2e-3;
    let e = |k: f64, atom: usize, c: usize| {
        solve(&displaced(&mol, atom, c, k * h), &bs, Reference::Rhf, fit)
            .scf
            .energy
    };
    let mut worst = 0.0f64;
    for (atom, c) in [(0usize, 0usize), (2, 2)] {
        let fd = (e(-2.0, atom, c) - 8.0 * e(-1.0, atom, c) + 8.0 * e(1.0, atom, c)
            - e(2.0, atom, c))
            / (12.0 * h);
        worst = worst.max((fd - g[(atom, c)]).abs());
    }
    eprintln!("HOI RI-JK ECP gradient vs FD of energy: {worst:.3e}");
    // ECP gradient components are O(1e-1); the fitted DF gradient is exact for
    // its energy, so only the FD truncation (h^4) and SCF convergence remain (measured 3.7e-10; the bar is 1e-7, far below any dropped ECP term).
    assert!(worst < 1e-7, "{worst:.3e}");
}

// ---------------------------------------------------------------------------
// Probe and dispatch
// ---------------------------------------------------------------------------

fn iodine(xyz: &str, mult: usize) -> (Molecule, PreparedBasis) {
    let bs = basis::bundled("def2-svp").unwrap();
    let mol = ecp_mol(xyz, mult, &bs);
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    (mol, prep)
}

#[test]
fn probe_accepts_ri_with_ecp_for_every_reference_where_the_metric_is_conditioned() {
    if !has_deriv2() {
        return;
    }
    let (mol, prep) = iodine(HOI, 1);
    let op = Operator::coulomb();
    for fit in [Fit::J(AUX_OK), Fit::K(AUX_OK), Fit::Jk(AUX_OK)] {
        let cfg = config(fit);
        ferric_scf::hessian::analytic_hessian_available(&mol, &prep, op, &cfg)
            .expect("RHF RI + ECP is analytic");
        analytic_uhf_hessian_available(&mol, &prep, op, &cfg).expect("UHF RI + ECP");
        analytic_rohf_hessian_available(&mol, &prep, op, &cfg).expect("ROHF RI + ECP");
    }
}

/// RI-K with the ordinary jkfit on iodine has cond 1e9 > RIK_COND_WARN: the
/// probe refuses (for the conditioning, not the ECP) and `Auto` runs FD.
#[test]
fn ill_conditioned_ri_k_with_ecp_is_refused_by_the_probe_and_auto_runs_fd() {
    if !has_deriv2() {
        return;
    }
    let (mol, prep) = iodine(HOI, 1);
    let cfg = config(Fit::Jk(AUX_HIGH_COND));
    let err =
        ferric_scf::hessian::analytic_hessian_available(&mol, &prep, Operator::coulomb(), &cfg)
            .expect_err("cond 1e9 RI-K must be refused by the probe");
    let msg = err.to_string();
    assert!(msg.contains("cond"), "{msg}");
    assert!(!msg.contains("effective core potential"), "{msg}");

    let bs = basis::bundled("def2-svp").unwrap();
    let mut hi = Molecule::parse_xyz("2\nHI\nH 0 0 0\nI 0 0 1.75\n", 0, 1).unwrap();
    hi.apply_ecp(&bs);
    let run = |hessian| {
        harmonic_frequencies(
            &ParallelContext::default(),
            &hi,
            "def2-svp",
            Operator::coulomb(),
            &config(Fit::Jk(AUX_HIGH_COND)),
            &FrequencyConfig {
                reference: FrequencyReference::Rhf,
                hessian,
                ..Default::default()
            },
        )
    };
    let auto = run(HessianMethod::Auto).expect("Auto falls back, never fails");
    assert_eq!(auto.hessian_source, HessianSource::FiniteDifference);
    assert_eq!(auto.n_gradient_evaluations, 12);
}

#[test]
fn ri_with_an_external_potential_stays_refused() {
    if !has_deriv2() {
        return;
    }
    let (mol, prep) = iodine(HOI, 1);
    let cfg = RhfConfig {
        external_potential: Some(ferric_core::external_potential::ExternalPotential {
            field: Some([0.0, 0.0, 0.004]),
            ..Default::default()
        }),
        ..config(Fit::J(AUX_OK))
    };
    let err =
        ferric_scf::hessian::analytic_hessian_available(&mol, &prep, Operator::coulomb(), &cfg)
            .expect_err("RI + environment stays refused");
    assert!(
        err.to_string()
            .contains("combined with an external potential"),
        "{err}"
    );
}
