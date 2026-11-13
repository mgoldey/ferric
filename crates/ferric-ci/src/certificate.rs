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

/// The warning of an unconverged CASSCF run.
pub fn casscf_unconverged_warning(r: &CasScfResult, cas: &CasScfConfig) -> String {
    format!(
        "CASSCF did not converge in {} macro iterations (|g| = {:.2e}, grad_tol {:e}, \
         energy_tol {:e})",
        r.iterations, r.grad_norm, cas.grad_tol, cas.energy_tol
    )
}

/// The certificate of `method.kind = "casscf"` for result `r` of config
/// `cas`.
pub fn casscf_certificate(r: &CasScfResult, cas: &CasScfConfig) -> Certificate {
    let c = Certificate::for_method_kind("casscf")
        .with_approximation("active-space", active_space_detail(cas));
    if r.converged {
        c
    } else {
        c.warn(casscf_unconverged_warning(r, cas))
            .cap_grade(Grade::Ungraded, "the CASSCF did not converge")
    }
}

/// The certificate of `method.kind = "nevpt2"` on reference `on` with
/// config `cas`.
pub fn nevpt2_certificate(on: Nevpt2On, cas: &CasScfConfig) -> Certificate {
    let c = Certificate::for_method_kind("nevpt2")
        .with_approximation("active-space", active_space_detail(cas))
        .with_approximation("reference", on.as_str());
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
