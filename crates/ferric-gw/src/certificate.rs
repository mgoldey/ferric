//! [`Certificate`](ferric_core::certificate::Certificate)s of GW, BSE-TDA and RPAx static-polarizability results:
//! capability and grade from `ferric_core::certificate::METHOD_KIND_GRADES`,
//! folded with the SCF reference ([`Certificate::for_post_scf`](ferric_core::certificate::Certificate::for_post_scf)), plus the
//! settings and convergence state of the run.
//!
//! The `gw` and `bse-tda` rows of `site/src/reference/validation.md` grade
//! agreement with PySCF at matched settings (`n_quad = 100`,
//! `trunc_thresh = 0`); a run at other settings carries a warning naming
//! its own. The build stamp comes from the reference certificate.

use ferric_core::certificate::{Certificate, Grade};
use ferric_rpa::config::PdepRpaConfig;

use crate::bse::BseResult;
use crate::{GwConfig, GwResult, UGwResult};

/// The quadrature size of the matched-settings PySCF comparison.
pub const MATCHED_N_QUAD: usize = 100;

/// Approximations and the matched-settings warning of the PDEP screening a
/// GW-type result is built on.
fn screening(c: Certificate, aux: &str, rpa: &PdepRpaConfig) -> Certificate {
    let q = &rpa.quadrature;
    let mut c = c
        .with_ri(aux)
        .with_frozen_core(rpa.frozen_core)
        .with_approximation(
            "frequency-quadrature",
            format!("{:?}, {} points", q.scheme, q.n_points).to_lowercase(),
        );
    if rpa.trunc_thresh > 0.0 {
        c = c.with_approximation(
            "pdep-truncation",
            format!("eigenpotentials below {:e} dropped", rpa.trunc_thresh),
        );
    }
    if q.n_points != MATCHED_N_QUAD || rpa.trunc_thresh != 0.0 {
        c.push_warning(format!(
            "agreement with PySCF is recorded at n_quad = {MATCHED_N_QUAD}, trunc_thresh = 0; \
             this run uses n_quad = {}, trunc_thresh = {:e}",
            q.n_points, rpa.trunc_thresh
        ));
    }
    c
}

/// Warnings for unconverged quasiparticle Newton roots, and the cap for an
/// unconverged eigenvalue self-consistency (evGW / evGW0).
fn gw_convergence(
    mut c: Certificate,
    gw: &GwConfig,
    mo_indices: &[usize],
    channels: &[(&str, &[bool])],
    n_ev_iter: usize,
    outer_converged: bool,
) -> Certificate {
    for (label, flags) in channels {
        let bad: Vec<usize> = mo_indices
            .iter()
            .zip(flags.iter())
            .filter(|(_, &ok)| !ok)
            .map(|(&m, _)| m)
            .collect();
        if !bad.is_empty() {
            c.push_warning(format!(
                "QP Newton solve did not converge for {label}MO(s) {bad:?}; those QP energies \
                 are best-effort"
            ));
        }
    }
    if outer_converged {
        c
    } else {
        c.cap_grade(
            Grade::Ungraded,
            format!(
                "{:?} eigenvalue self-consistency did not converge in {n_ev_iter} iterations \
                 (thresh {:e})",
                gw.method, gw.ev_conv_thresh
            ),
        )
    }
}

fn gw_base(reference: &Certificate, aux: &str, rpa: &PdepRpaConfig, gw: &GwConfig) -> Certificate {
    let mut c = screening(Certificate::for_post_scf("gw", reference), aux, rpa)
        .with_approximation("gw-method", format!("{:?}", gw.method));
    if gw.pade_npts > 0 {
        c = c.with_approximation(
            "analytic-continuation",
            format!("Pade on {} points", gw.pade_npts),
        );
    }
    c
}

/// Closed-shell GW (`gw`).
pub fn gw_certificate(
    reference: &Certificate,
    aux: &str,
    rpa: &PdepRpaConfig,
    gw: &GwConfig,
    r: &GwResult,
) -> Certificate {
    gw_convergence(
        gw_base(reference, aux, rpa, gw),
        gw,
        &r.mo_indices,
        &[("", &r.qp_converged)],
        r.n_ev_iter,
        r.outer_converged,
    )
}

/// Open-shell GW (`gw`, multiplicity > 1).
pub fn u_gw_certificate(
    reference: &Certificate,
    aux: &str,
    rpa: &PdepRpaConfig,
    gw: &GwConfig,
    r: &UGwResult,
) -> Certificate {
    gw_convergence(
        gw_base(reference, aux, rpa, gw),
        gw,
        &r.mo_indices,
        &[("alpha ", &r.qp_converged_a), ("beta ", &r.qp_converged_b)],
        r.n_ev_iter,
        r.outer_converged,
    )
}

/// BSE-TDA (`bse-tda`) on G0W0 quasiparticle energies.
pub fn bse_tda_certificate(
    reference: &Certificate,
    aux: &str,
    rpa: &PdepRpaConfig,
    r: &BseResult,
) -> Certificate {
    let mut c = screening(Certificate::for_post_scf("bse-tda", reference), aux, rpa)
        .with_approximation("tda", "Tamm-Dancoff, statically screened kernel");
    if !r.qp_suspect_mos.is_empty() {
        c.push_warning(format!(
            "quasiparticle energies of MO(s) {:?} are suspect (QP solve flagged them)",
            r.qp_suspect_mos
        ));
    }
    c
}

/// Statically screened RPAx polarizability (`tdhf-static-polarizability`).
/// `scissor` is the `[gw] scissor` shift (Ha) applied to the gaps.
pub fn tdhf_static_polarizability_certificate(
    reference: &Certificate,
    aux: &str,
    rpa: &PdepRpaConfig,
    scissor: f64,
) -> Certificate {
    let mut c = Certificate::for_post_scf("tdhf-static-polarizability", reference)
        .with_ri(aux)
        .with_frozen_core(rpa.frozen_core)
        .with_approximation("kernel", "statically screened RPAx");
    if rpa.trunc_thresh > 0.0 {
        c = c.with_approximation(
            "pdep-truncation",
            format!("eigenpotentials below {:e} dropped", rpa.trunc_thresh),
        );
    }
    if scissor != 0.0 {
        c = c.with_approximation("scissor", format!("{scissor} Ha gap shift"));
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_rpa::config::QuadratureConfig;

    fn rhf() -> Certificate {
        Certificate::for_method_kind("rhf")
    }

    fn rpa(n_points: usize, trunc: f64) -> PdepRpaConfig {
        PdepRpaConfig {
            trunc_thresh: trunc,
            quadrature: QuadratureConfig {
                n_points,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn names(c: &Certificate) -> Vec<&str> {
        c.approximations.iter().map(|a| a.name.as_str()).collect()
    }

    fn base_gw() -> Certificate {
        gw_base(&rhf(), "cc-pvdz-ri", &rpa(20, 1e-4), &GwConfig::default())
    }

    #[test]
    fn gw_grade_settings_and_matched_settings_warning() {
        let c = base_gw();
        assert_eq!(c.capability, "gw");
        assert_eq!(c.grade, Grade::Smoke);
        assert_eq!(
            names(&c),
            ["ri", "frequency-quadrature", "pdep-truncation", "gw-method"]
        );
        assert_eq!(c.approximations[3].detail, "G0W0");
        assert_eq!(c.warnings.len(), 1);
        assert!(c.warnings[0].contains("this run uses n_quad = 20, trunc_thresh = 1e-4"));
        // Either setting alone off the matched pair warns.
        for (n, t) in [(100, 1e-4), (40, 0.0)] {
            let one = gw_base(&rhf(), "a", &rpa(n, t), &GwConfig::default());
            assert_eq!(one.warnings.len(), 1, "n_quad = {n}, trunc_thresh = {t}");
        }
        // At the matched settings the caveat is absent.
        let m = gw_base(&rhf(), "a", &rpa(100, 0.0), &GwConfig::default());
        assert!(m.warnings.is_empty(), "{:?}", m.warnings);
        assert_eq!(names(&m), ["ri", "frequency-quadrature", "gw-method"]);
    }

    #[test]
    fn gw_convergence_failures_warn_and_cap() {
        let gw = GwConfig::default();
        let ok = gw_convergence(base_gw(), &gw, &[3, 4], &[("", &[true, true])], 1, true);
        assert_eq!(ok.grade, Grade::Smoke);
        let qp = gw_convergence(
            base_gw(),
            &gw,
            &[3, 4],
            &[("alpha ", &[true, false]), ("beta ", &[true, true])],
            1,
            true,
        );
        assert_eq!(qp.grade, Grade::Smoke, "a QP root failure warns only");
        assert!(qp
            .warnings
            .iter()
            .any(|w| w.starts_with("QP Newton solve did not converge for alpha MO(s) [4]")));
        assert!(!qp.warnings.iter().any(|w| w.contains("beta MO")));
        let outer = gw_convergence(base_gw(), &gw, &[3], &[("", &[true])], 20, false);
        assert_eq!(outer.grade, Grade::Ungraded);
        assert!(outer
            .warnings
            .iter()
            .any(|w| w.contains("self-consistency did not converge in 20 iterations")));
    }

    #[test]
    fn bse_and_rpax_grades_and_settings() {
        let r = BseResult {
            omega: vec![],
            nocc: 1,
            nvir: 1,
            eps_qp: vec![],
            oscillator_strength: vec![],
            qp_suspect_mos: vec![2],
        };
        let b = bse_tda_certificate(&rhf(), "a", &rpa(100, 0.0), &r);
        assert_eq!(b.capability, "bse-tda");
        assert_eq!(b.grade, Grade::Smoke);
        assert_eq!(b.warnings.len(), 1);
        assert!(b.warnings[0].contains("MO(s) [2] are suspect"));
        let t = tdhf_static_polarizability_certificate(&rhf(), "a", &rpa(20, 1e-4), 0.36);
        assert_eq!(t.capability, "tdhf-static-polarizability");
        assert_eq!(t.grade, Grade::Smoke);
        assert_eq!(names(&t), ["ri", "kernel", "pdep-truncation", "scissor"]);
    }
}
