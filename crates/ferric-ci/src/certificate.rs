//! Certificates of the CASSCF and SC-NEVPT2 results, shared by the CLI kinds
//! `casscf` / `nevpt2` and the Python `run_casscf` / `run_nevpt2`.
//!
//! The grade is the kind's row of
//! [`METHOD_KIND_GRADES`](ferric_core::certificate::METHOD_KIND_GRADES). An
//! unconverged CASSCF is capped to not graded and warns; SC-NEVPT2 on a
//! CAS-CI reference is capped to not graded (its PySCF comparison runs on a
//! CASSCF reference only). The active space is recorded as an
//! approximation. The caller stamps the build.

use crate::driver::Nevpt2On;
use crate::{CasScfConfig, CasScfResult};
use ferric_core::certificate::{Certificate, Grade};
use ferric_mp2::certificate::with_anchor_in_scope;

/// `CAS(n_el, n_act)` with the sector and the core count.
fn active_space_detail(cas: &CasScfConfig) -> String {
    let (na, nb) = cas.n_elec_active;
    format!(
        "CAS({}, {}): {na} alpha and {nb} beta electrons in {} orbitals above {} core",
        na + nb,
        cas.n_active,
        cas.n_active,
        cas.n_core
    )
}

/// The warning of an unconverged CASSCF run, naming the TOML keys.
pub fn casscf_unconverged_warning(r: &CasScfResult, cas: &CasScfConfig) -> String {
    casscf_unconverged_warning_prefixed(r, cas, "")
}

/// The warning of an unconverged CASSCF run with every tolerance named by
/// `prefix` + its config key: the Python entry points pass `"casscf_"` so the
/// text names the keyword the caller wrote (`casscf_grad_tol`), not the
/// library field.
pub fn casscf_unconverged_warning_prefixed(
    r: &CasScfResult,
    cas: &CasScfConfig,
    prefix: &str,
) -> String {
    format!(
        "CASSCF did not converge in {} macro iterations (|g| = {:.2e}, {prefix}grad_tol {:e}, \
         {prefix}energy_tol {:e})",
        r.iterations, r.grad_norm, cas.grad_tol, cas.energy_tol
    )
}

/// Whether `cas` is what the energy anchors' evidence runs: canonical natural
/// orbitals, and a gradient and energy tolerance no looser than the default
/// the evidence runs (a looser one leaves a larger orbital error than the
/// anchor records).
fn in_casscf_anchor_scope(cas: &CasScfConfig) -> bool {
    cas.canonicalize && in_casscf_tolerances(cas)
}

/// Whether `cas` converges the orbitals as tightly as the evidence runs. The
/// SC-NEVPT2 anchor needs only this: the driver canonicalizes the core and
/// virtual orbitals itself and the energy does not depend on the gauge of the
/// active orbitals.
fn in_casscf_tolerances(cas: &CasScfConfig) -> bool {
    let d = CasScfConfig::default();
    cas.grad_tol <= d.grad_tol && cas.energy_tol <= d.energy_tol
}

/// The certificate of `method.kind = "casscf"` for result `r` of config
/// `cas`. A converged run in the scope of `casscf_vs_pyscf.rs` (see
/// `in_casscf_anchor_scope`) carries the CASSCF energy anchor.
pub fn casscf_certificate(r: &CasScfResult, cas: &CasScfConfig) -> Certificate {
    casscf_certificate_prefixed(r, cas, "")
}

/// [`casscf_certificate`] whose warning names the tolerances with `prefix`
/// (see [`casscf_unconverged_warning_prefixed`]).
pub fn casscf_certificate_prefixed(
    r: &CasScfResult,
    cas: &CasScfConfig,
    prefix: &str,
) -> Certificate {
    let c = Certificate::for_method_kind("casscf")
        .with_approximation("active-space", active_space_detail(cas));
    let c = with_anchor_in_scope(c, "CASSCF", r.converged && in_casscf_anchor_scope(cas));
    if r.converged {
        c
    } else {
        c.warn(casscf_unconverged_warning_prefixed(r, cas, prefix))
            .cap_grade(Grade::Ungraded, "the CASSCF did not converge")
    }
}

/// The certificate of `method.kind = "nevpt2"` on reference `on` with
/// config `cas`. SC-NEVPT2 on a CASSCF reference with a closed-shell active
/// space converged to the CASSCF evidence's tolerances carries the SC-NEVPT2 anchor
/// (`nevpt2.rs` compares class energies with PySCF on three singlets).
pub fn nevpt2_certificate(on: Nevpt2On, cas: &CasScfConfig) -> Certificate {
    let c = Certificate::for_method_kind("nevpt2")
        .with_approximation("active-space", active_space_detail(cas))
        .with_approximation("reference", on.as_str());
    let c = with_anchor_in_scope(
        c,
        "SC-NEVPT2",
        on == Nevpt2On::Casscf
            && cas.n_elec_active.0 == cas.n_elec_active.1
            && in_casscf_tolerances(cas),
    );
    match on {
        Nevpt2On::Casscf => c,
        Nevpt2On::Casci => c.cap_grade(
            Grade::Ungraded,
            "SC-NEVPT2 on a CAS-CI reference is not compared against PySCF",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CasCiRdms;
    use ndarray::{Array2, Array4};

    fn cas() -> CasScfConfig {
        CasScfConfig {
            n_active: 6,
            n_elec_active: (5, 3),
            n_core: 4,
            ..Default::default()
        }
    }

    fn result(converged: bool) -> CasScfResult {
        CasScfResult {
            e_total: -149.6,
            e_core: -100.0,
            mo_coeff: Array2::zeros((1, 1)),
            mo_energy: None,
            ci_vector: vec![1.0],
            rdms: CasCiRdms {
                dm1: Array2::zeros((1, 1)),
                dm2: Array4::zeros((1, 1, 1, 1)),
            },
            n_determinants: 1,
            iterations: 300,
            grad_norm: 3.0e-5,
            converged,
            ci_residual_norm: 0.0,
        }
    }

    #[test]
    fn a_converged_casscf_carries_the_kind_grade_and_the_active_space() {
        let c = casscf_certificate(&result(true), &cas());
        assert_eq!(c.capability, "casscf");
        assert_eq!(c.grade, Grade::Proven);
        assert!(c.warnings.is_empty(), "{:?}", c.warnings);
        assert_eq!(c.approximations.len(), 1);
        assert_eq!(c.approximations[0].name, "active-space");
        assert_eq!(
            c.approximations[0].detail,
            "CAS(8, 6): 5 alpha and 3 beta electrons in 6 orbitals above 4 core"
        );
    }

    #[test]
    fn casscf_anchor_is_attached_inside_the_evidence_scope_only() {
        let canonical = CasScfConfig {
            canonicalize: true,
            ..cas()
        };
        let c = casscf_certificate(&result(true), &canonical);
        let m = c.measured_error.as_ref().expect("anchor attached");
        assert_eq!(
            *m,
            ferric_core::certificate::energy_anchor("CASSCF")
                .unwrap()
                .measured_error()
        );
        assert_eq!(m.evidence, "crates/ferric-ci/tests/casscf_vs_pyscf.rs");
        let d = CasScfConfig::default();
        let outside = [
            (
                "not canonical",
                CasScfConfig {
                    canonicalize: false,
                    ..canonical
                },
                true,
            ),
            (
                "loose gradient",
                CasScfConfig {
                    grad_tol: d.grad_tol * 10.0,
                    ..canonical
                },
                true,
            ),
            (
                "loose energy",
                CasScfConfig {
                    energy_tol: d.energy_tol * 10.0,
                    ..canonical
                },
                true,
            ),
            ("unconverged", canonical, false),
        ];
        for (what, cfg, converged) in outside {
            let c = casscf_certificate(&result(converged), &cfg);
            assert!(c.measured_error.is_none(), "{what}: anchor attached");
        }
    }

    #[test]
    fn nevpt2_anchor_is_attached_inside_the_evidence_scope_only() {
        let singlet = CasScfConfig {
            n_elec_active: (3, 3),
            canonicalize: false,
            ..cas()
        };
        let c = nevpt2_certificate(Nevpt2On::Casscf, &singlet);
        let m = c.measured_error.as_ref().expect("anchor attached");
        assert_eq!(
            *m,
            ferric_core::certificate::energy_anchor("SC-NEVPT2")
                .unwrap()
                .measured_error()
        );
        assert_eq!(m.evidence, "crates/ferric-ci/tests/nevpt2.rs");
        let outside = [
            ("casci reference", Nevpt2On::Casci, singlet),
            (
                "open-shell active space",
                Nevpt2On::Casscf,
                CasScfConfig {
                    n_elec_active: (4, 2),
                    ..singlet
                },
            ),
            (
                "loose gradient",
                Nevpt2On::Casscf,
                CasScfConfig {
                    grad_tol: CasScfConfig::default().grad_tol * 10.0,
                    ..singlet
                },
            ),
        ];
        for (what, on, cfg) in outside {
            let c = nevpt2_certificate(on, &cfg);
            assert!(c.measured_error.is_none(), "{what}: anchor attached");
        }
    }

    #[test]
    fn an_unconverged_casscf_is_capped_and_warns() {
        let c = casscf_certificate(&result(false), &cas());
        assert_eq!(c.grade, Grade::Ungraded);
        assert_eq!(
            c.warnings,
            vec![
                casscf_unconverged_warning(&result(false), &cas()),
                "grade capped from Proven to not graded: the CASSCF did not converge".to_string(),
            ]
        );
        assert!(c.warnings[0].starts_with("CASSCF did not converge in 300 macro iterations"));
    }

    #[test]
    fn nevpt2_is_graded_on_casscf_and_capped_on_casci() {
        let on_casscf = nevpt2_certificate(Nevpt2On::Casscf, &cas());
        assert_eq!(on_casscf.capability, "nevpt2");
        assert_eq!(on_casscf.grade, Grade::Proven);
        let names: Vec<&str> = on_casscf
            .approximations
            .iter()
            .map(|a| a.name.as_str())
            .collect();
        assert_eq!(names, ["active-space", "reference"]);
        assert_eq!(on_casscf.approximations[1].detail, "casscf");
        let on_casci = nevpt2_certificate(Nevpt2On::Casci, &cas());
        assert_eq!(on_casci.grade, Grade::Ungraded);
        assert_eq!(on_casci.approximations[1].detail, "casci");
    }
}
