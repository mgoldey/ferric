//! [`Certificate`](ferric_core::certificate::Certificate)s of PDEP-RPA and range-separated MP2+RPA results:
//! capability and grade from `ferric_core::certificate::METHOD_KIND_GRADES`,
//! folded with the SCF reference ([`Certificate::for_post_scf`](ferric_core::certificate::Certificate::for_post_scf)), plus the
//! settings of the run. The build stamp comes from the reference.

use ferric_core::certificate::Certificate;
use ferric_mp2::certificate::with_anchor_in_scope;

use crate::config::{Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig, QuadratureScheme};
use crate::rs_mp2_rpa::Attenuator;
use crate::{PdepRpaResult, RsMp2RpaConfig};

/// The quadrature sizes `validation_rpa_gradient.rs` compares with PySCF's
/// `gw.rpa.RPA`: the 40-point grid and the matched 6-point control.
const EVIDENCE_N_POINTS: [usize; 2] = [6, 40];

/// Whether `cfg` on `reference` is what the PDEP-RPA anchor's evidence runs: an
/// RHF reference, all electrons correlated, every dielectric mode kept
/// (`trunc_thresh <= 0`, the dense `eigh` of the Lanczos arm), the dense
/// χ₀, and Gauss-Legendre points at u₀ = 0.5 on the grids of
/// `EVIDENCE_N_POINTS`.
fn in_pdep_anchor_scope(reference: &Certificate, cfg: &PdepRpaConfig) -> bool {
    let q = &cfg.quadrature;
    reference.capability == "rhf"
        && cfg.frozen_core == 0
        && cfg.trunc_thresh <= 0.0
        && cfg.eigensolver == Eigensolver::Lanczos
        && cfg.chi0_backend == Chi0Backend::Dense
        && cfg.chi0_sparsity == Chi0Sparsity::Dense
        && q.scheme == QuadratureScheme::GaussLegendre
        && q.u0 == 0.5
        && EVIDENCE_N_POINTS.contains(&q.n_points)
}

/// PDEP-RPA (`pdep-rpa`) run with `cfg`. The PDEP-RPA energy anchor is
/// attached inside the scope of its evidence (see `in_pdep_anchor_scope`).
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
    c = with_anchor_in_scope(c, "PDEP-RPA", in_pdep_anchor_scope(reference, cfg));
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

    /// Each setting outside the evidence's drops the anchor on its own.
    #[test]
    fn pdep_rpa_anchor_is_attached_inside_the_evidence_scope_only() {
        let rhf = Certificate::for_method_kind("rhf");
        let base = || PdepRpaConfig {
            trunc_thresh: 0.0,
            quadrature: crate::config::QuadratureConfig {
                n_points: 40,
                ..Default::default()
            },
            ..Default::default()
        };
        let cert = |reference: &Certificate, cfg: &PdepRpaConfig| {
            pdep_rpa_certificate(reference, "cc-pvdz-ri", cfg, &result(true))
        };
        let m = cert(&rhf, &base()).measured_error.expect("anchor attached");
        assert_eq!(
            m,
            ferric_core::certificate::energy_anchor("PDEP-RPA")
                .unwrap()
                .measured_error()
        );
        assert_eq!(
            m.evidence,
            "crates/ferric-rpa/tests/validation_rpa_gradient.rs"
        );
        let mut coarse = base();
        coarse.quadrature.n_points = 6;
        assert!(cert(&rhf, &coarse).measured_error.is_some());
        let with = |f: &dyn Fn(&mut PdepRpaConfig)| {
            let mut c = base();
            f(&mut c);
            c
        };
        let outside = [
            ("frozen core", with(&|c| c.frozen_core = 1)),
            ("truncation", with(&|c| c.trunc_thresh = 1e-4)),
            ("n_points", with(&|c| c.quadrature.n_points = 20)),
            ("u0", with(&|c| c.quadrature.u0 = 0.7)),
            (
                "scheme",
                with(&|c| c.quadrature.scheme = QuadratureScheme::MiniMax),
            ),
            ("davidson", with(&|c| c.eigensolver = Eigensolver::Davidson)),
            (
                "laplace chi0",
                with(&|c| c.chi0_backend = Chi0Backend::Laplace { n_quad: 7 }),
            ),
        ];
        for (what, cfg) in outside {
            assert!(
                cert(&rhf, &cfg).measured_error.is_none(),
                "{what}: anchor attached"
            );
        }
        let uhf = Certificate::for_method_kind("uhf");
        assert!(cert(&uhf, &base()).measured_error.is_none());
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
