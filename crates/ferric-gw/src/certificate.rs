//! [`Certificate`](ferric_core::certificate::Certificate)s of GW, BSE-TDA and RPAx static-polarizability results:
//! capability and grade from `ferric_core::certificate::METHOD_KIND_GRADES`,
//! folded with the SCF reference ([`Certificate::for_post_scf`](ferric_core::certificate::Certificate::for_post_scf)), plus the
//! settings and convergence state of the run.
//!
//! The `bse-tda` row of `site/src/reference/validation.md` grades agreement
//! with PySCF at matched settings (`n_quad = 100`, `trunc_thresh = 0`); a run
//! at other settings carries a warning naming its own. The `gw` row also
//! grades the measured error of the GW defaults against those settings, so a
//! `gw` run warns only outside that calibrated envelope (Gauss–Legendre,
//! u₀ = 0.5, `n_quad ≥ DEFAULT_N_QUAD`, `trunc_thresh ≤ 1e-4`). The build
//! stamp comes from the reference certificate.

use ferric_core::certificate::{Certificate, Grade};
use ferric_rpa::config::{PdepRpaConfig, QuadratureScheme};

use crate::bse::BseResult;
use crate::{GwConfig, GwMethod, GwResult, UGwResult};

/// The quadrature size of the matched-settings PySCF comparison.
pub const MATCHED_N_QUAD: usize = 100;

/// The largest PDEP truncation inside GW's calibrated envelope (the default).
pub const CALIBRATED_TRUNC_THRESH: f64 = 1e-4;

/// The settings envelope a screening certificate is graded on.
#[derive(Clone, Copy)]
enum Envelope {
    /// Only the matched PySCF settings (`bse-tda`).
    Matched,
    /// GW: Gauss–Legendre with u₀ = 0.5, `n_quad ≥ DEFAULT_N_QUAD`,
    /// `trunc_thresh ≤ CALIBRATED_TRUNC_THRESH`. Inside it the error
    /// against the matched settings is measured at ≤ 10 meV on HOMO/LUMO
    /// (`crate::DEFAULT_N_QUAD`).
    GwCalibrated,
}

/// Approximations and the settings warning of the PDEP screening a GW-type
/// result is built on.
fn screening(c: Certificate, aux: &str, rpa: &PdepRpaConfig, env: Envelope) -> Certificate {
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
    match env {
        Envelope::Matched => {
            if q.n_points != MATCHED_N_QUAD || rpa.trunc_thresh != 0.0 {
                c.push_warning(format!(
                    "agreement with PySCF is recorded at n_quad = {MATCHED_N_QUAD}, \
                     trunc_thresh = 0; this run uses n_quad = {}, trunc_thresh = {:e}",
                    q.n_points, rpa.trunc_thresh
                ));
            }
        }
        Envelope::GwCalibrated => {
            let inside = q.scheme == QuadratureScheme::GaussLegendre
                && q.u0 == 0.5
                && q.n_points >= crate::DEFAULT_N_QUAD
                && rpa.trunc_thresh <= CALIBRATED_TRUNC_THRESH;
            if !inside {
                c.push_warning(format!(
                    "the error against the PySCF-matched settings (n_quad = {MATCHED_N_QUAD}, \
                     trunc_thresh = 0) is measured at <= 10 meV on HOMO/LUMO only for \
                     Gauss-Legendre (u0 = 0.5) with n_quad >= {}, trunc_thresh <= {:e}; this \
                     run uses {:?}, u0 = {}, n_quad = {}, trunc_thresh = {:e}",
                    crate::DEFAULT_N_QUAD,
                    CALIBRATED_TRUNC_THRESH,
                    q.scheme,
                    q.u0,
                    q.n_points,
                    rpa.trunc_thresh
                ));
            }
        }
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
    let mut c = screening(
        Certificate::for_post_scf("gw", reference),
        aux,
        rpa,
        Envelope::GwCalibrated,
    )
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

/// The warning of an open-shell GW run on a KS reference outside what is
/// compared: a ROKS reference, or eigenvalue self-consistency (U-evGW₀,
/// U-evGW) or U-COHSEX on a UKS reference. U-G0W0 on a UKS reference is
/// compared against the grid-free U-dRPA pole sum (OH, CH3, NH2 / cc-pVDZ)
/// and converged in n_quad at the default (`tests/validation_gw_defaults.rs`),
/// so it carries no warning.
pub const OPEN_SHELL_KS_WARNING: &str =
    "U-GW on an open-shell KS reference is compared only for U-G0W0 on a UKS reference \
     (OH, CH3, NH2 / cc-pVDZ, against the grid-free U-dRPA pole sum); a ROKS reference and \
     U-evGW0 / U-evGW / U-COHSEX on a KS reference are not compared";

/// Open-shell GW (`gw`, multiplicity > 1). `ks_reference` is whether the
/// reference is UKS or ROKS (a v_xc diagonal entered the QP equation); the
/// reference certificate's capability (`uhf` / `rohf`) tells the two apart.
pub fn u_gw_certificate(
    reference: &Certificate,
    aux: &str,
    rpa: &PdepRpaConfig,
    gw: &GwConfig,
    r: &UGwResult,
    ks_reference: bool,
) -> Certificate {
    let mut base = gw_base(reference, aux, rpa, gw);
    if ks_reference && (reference.capability != "uhf" || gw.method != GwMethod::G0W0) {
        base.push_warning(OPEN_SHELL_KS_WARNING);
    }
    gw_convergence(
        base,
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
    let mut c = screening(
        Certificate::for_post_scf("bse-tda", reference),
        aux,
        rpa,
        Envelope::Matched,
    )
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

    /// The CLI / Python GW defaults.
    fn base_gw() -> Certificate {
        gw_base(
            &rhf(),
            "cc-pvdz-ri",
            &rpa(crate::DEFAULT_N_QUAD, CALIBRATED_TRUNC_THRESH),
            &GwConfig::default(),
        )
    }

    #[test]
    fn gw_grade_settings_and_calibrated_envelope_warning() {
        let c = base_gw();
        assert_eq!(c.capability, "gw");
        assert_eq!(c.grade, Grade::ProvenNarrow);
        assert_eq!(
            names(&c),
            ["ri", "frequency-quadrature", "pdep-truncation", "gw-method"]
        );
        assert_eq!(c.approximations[3].detail, "G0W0");
        // The defaults are inside the calibrated envelope: no caveat.
        assert!(c.warnings.is_empty(), "{:?}", c.warnings);
        // Inside the envelope, matched settings included: no caveat.
        for (n, t) in [(100, 0.0), (40, 0.0), (60, 1e-5)] {
            let m = gw_base(&rhf(), "a", &rpa(n, t), &GwConfig::default());
            assert!(
                m.warnings.is_empty(),
                "n_quad = {n}, trunc = {t}: {:?}",
                m.warnings
            );
        }
        let m = gw_base(&rhf(), "a", &rpa(100, 0.0), &GwConfig::default());
        assert_eq!(names(&m), ["ri", "frequency-quadrature", "gw-method"]);
        // Each setting alone outside the envelope warns, naming the run's.
        let coarse = gw_base(&rhf(), "a", &rpa(20, 1e-4), &GwConfig::default());
        assert_eq!(coarse.warnings.len(), 1);
        assert!(coarse.warnings[0].contains("n_quad = 20, trunc_thresh = 1e-4"));
        for (n, t) in [(39, 0.0), (100, 1e-3)] {
            let one = gw_base(&rhf(), "a", &rpa(n, t), &GwConfig::default());
            assert_eq!(one.warnings.len(), 1, "n_quad = {n}, trunc_thresh = {t}");
        }
        let mut mm = rpa(100, 0.0);
        mm.quadrature.scheme = QuadratureScheme::MiniMax;
        assert_eq!(
            gw_base(&rhf(), "a", &mm, &GwConfig::default())
                .warnings
                .len(),
            1
        );
        let mut u0 = rpa(100, 0.0);
        u0.quadrature.u0 = 1.0;
        assert_eq!(
            gw_base(&rhf(), "a", &u0, &GwConfig::default())
                .warnings
                .len(),
            1
        );
    }

    #[test]
    fn gw_convergence_failures_warn_and_cap() {
        let gw = GwConfig::default();
        let ok = gw_convergence(base_gw(), &gw, &[3, 4], &[("", &[true, true])], 1, true);
        assert_eq!(ok.grade, Grade::ProvenNarrow);
        let qp = gw_convergence(
            base_gw(),
            &gw,
            &[3, 4],
            &[("alpha ", &[true, false]), ("beta ", &[true, true])],
            1,
            true,
        );
        assert_eq!(
            qp.grade,
            Grade::ProvenNarrow,
            "a QP root failure warns only"
        );
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
        // BSE-TDA is graded at the matched settings only: the GW defaults
        // still warn there.
        let r_ok = BseResult {
            qp_suspect_mos: vec![],
            ..r
        };
        let bd = bse_tda_certificate(
            &rhf(),
            "a",
            &rpa(crate::DEFAULT_N_QUAD, CALIBRATED_TRUNC_THRESH),
            &r_ok,
        );
        assert_eq!(bd.warnings.len(), 1);
        assert!(bd.warnings[0].contains("agreement with PySCF is recorded at n_quad = 100"));
        let t = tdhf_static_polarizability_certificate(&rhf(), "a", &rpa(20, 1e-4), 0.36);
        assert_eq!(t.capability, "tdhf-static-polarizability");
        assert_eq!(t.grade, Grade::Smoke);
        assert_eq!(names(&t), ["ri", "kernel", "pdep-truncation", "scissor"]);
    }
}
