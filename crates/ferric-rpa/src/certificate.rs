//! [`Certificate`](ferric_core::certificate::Certificate)s of PDEP-RPA and range-separated MP2+RPA results:
//! capability and grade from `ferric_core::certificate::METHOD_KIND_GRADES`,
//! folded with the SCF reference ([`Certificate::for_post_scf`](ferric_core::certificate::Certificate::for_post_scf)), plus the
//! settings of the run. The build stamp comes from the reference.

use ferric_core::certificate::Certificate;

use crate::config::PdepRpaConfig;
use crate::rs_mp2_rpa::Attenuator;
use crate::{PdepRpaResult, RsMp2RpaConfig};

/// PDEP-RPA (`pdep-rpa`) run with `cfg`.
pub fn pdep_rpa_certificate(
    reference: &Certificate,
    aux: &str,
    cfg: &PdepRpaConfig,
    r: &PdepRpaResult,
) -> Certificate {
    let q = &cfg.quadrature;
    let mut c = Certificate::for_post_scf("pdep-rpa", reference)
        .with_ri(aux)
        .with_frozen_core(cfg.frozen_core)
        .with_approximation(
            "frequency-quadrature",
            format!("{:?}, {} points", q.scheme, q.n_points).to_lowercase(),
        );
    if cfg.trunc_thresh > 0.0 {
        c = c.with_approximation(
            "pdep-truncation",
            format!(
                "eigenpotentials below {:e} dropped, {} of {} kept",
                cfg.trunc_thresh,
                r.n_eigenpotentials,
                r.eigenvalues_static.len()
            ),
        );
    }
    if !r.eigensolver_converged {
        c.push_warning(
            "PDEP eigensolver did not fully converge: best-effort Ritz pairs, not verified to \
             residual tolerance",
        );
    }
    c
}

/// Range-separated MP2 + long-range RPA (`rs-mp2-rpa`) run with `cfg`.
pub fn rs_mp2_rpa_certificate(
    reference: &Certificate,
    aux: &str,
    cfg: &RsMp2RpaConfig,
) -> Certificate {
    let split = match cfg.attenuator {
        Attenuator::Erf => format!("erf/erfc split, omega = {} Bohr^-1", cfg.omega),
        Attenuator::Terf => format!("terf/terfc split, r0 = {} Bohr", cfg.r0),
    };
    let q = &cfg.drpa.quadrature;
    Certificate::for_post_scf("rs-mp2-rpa", reference)
        .with_ri(aux)
        .with_frozen_core(cfg.frozen_core)
        .with_approximation(
            "range-separation",
            format!("{split}, {:?}", cfg.formulation),
        )
        .with_approximation(
            "frequency-quadrature",
            format!("{:?}, {} points", q.scheme, q.n_points).to_lowercase(),
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::certificate::Grade;

    fn result(converged: bool) -> PdepRpaResult {
        PdepRpaResult {
            e_rpa: -0.1,
            n_eigenpotentials: 3,
            eigenvalues_static: vec![0.0; 5],
            eigenpotentials: Default::default(),
            dressed_eigenvectors: Default::default(),
            quad_freqs: vec![],
            quad_weights: vec![],
            eigenvalues_freq: Default::default(),
            inv_dielectric_freq: None,
            e_rpa_dft_diag: None,
            eigensolver_converged: converged,
        }
    }

    #[test]
    fn pdep_rpa_grade_settings_and_eigensolver_warning() {
        let rhf = Certificate::for_method_kind("rhf");
        let cfg = PdepRpaConfig {
            trunc_thresh: 1e-4,
            ..Default::default()
        };
        let c = pdep_rpa_certificate(&rhf, "cc-pvdz-ri", &cfg, &result(true));
        assert_eq!(c.capability, "pdep-rpa");
        assert_eq!(c.grade, Grade::Proven);
        let names: Vec<&str> = c.approximations.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["ri", "frequency-quadrature", "pdep-truncation"]);
        assert!(c.approximations[2].detail.ends_with("3 of 5 kept"));
        assert!(c.warnings.is_empty());
        let full = PdepRpaConfig {
            trunc_thresh: 0.0,
            ..Default::default()
        };
        let u = pdep_rpa_certificate(&rhf, "a", &full, &result(false));
        assert_eq!(u.approximations.len(), 2, "full rank: no truncation entry");
        assert_eq!(u.warnings.len(), 1);
        assert!(u.warnings[0].starts_with("PDEP eigensolver did not fully converge"));
    }

    #[test]
    fn rs_mp2_rpa_is_smoke_with_its_split() {
        let cfg = RsMp2RpaConfig {
            omega: 0.2,
            attenuator: Attenuator::Erf,
            r0: 3.0,
            terf_omega: None,
            frozen_core: 0,
            drpa: PdepRpaConfig::default(),
            formulation: crate::RsMp2RpaFormulation::DeltaLr,
        };
        let c = rs_mp2_rpa_certificate(&Certificate::for_method_kind("rhf"), "a", &cfg);
        assert_eq!(c.grade, Grade::Smoke);
        assert_eq!(
            c.approximations[1].detail,
            "erf/erfc split, omega = 0.2 Bohr^-1, DeltaLr"
        );
    }
}
