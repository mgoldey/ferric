//! [`Certificate`](ferric_core::certificate::Certificate)s of the MP2-family results: capability and grade from
//! `ferric_core::certificate::METHOD_KIND_GRADES`, folded with the SCF
//! reference's certificate ([`Certificate::for_post_scf`](ferric_core::certificate::Certificate::for_post_scf)), plus the
//! settings each method ran with.
//!
//! Built after the solve from the config the method ran with; nothing here
//! changes an energy. The build stamp comes from the reference certificate
//! (`ferric-cli` and `ferric-python` stamp it).

use ferric_core::certificate::{Certificate, Grade};
use ferric_core::gpu::provenance::PrecisionRecord;
use ferric_integrals::operator::{Operator, OperatorKind};

use crate::attenuated::AttenuatedMp2Config;
use crate::double_hybrid::Mp2DoubleHybridConfig;
use crate::laplace::{SosFormulation, SosMp2Config};
use crate::lmp2_certificate::{operator_label, Lmp2Certificate};
use crate::rimp2::RiMp2Config;
use crate::scs::{ScsMp2Config, ScsMp2TerfcConfig};

/// The certificate every RI correlation method shares: `kind` on
/// `reference`, the auxiliary basis, frozen core, and the correlation
/// operator when it is not Coulomb.
pub fn ri_correlation_certificate(
    kind: &str,
    reference: &Certificate,
    aux: &str,
    frozen_core: usize,
    op: &Operator,
) -> Certificate {
    let c = Certificate::for_post_scf(kind, reference)
        .with_ri(aux)
        .with_frozen_core(frozen_core);
    if op.kind == OperatorKind::Coulomb {
        c
    } else {
        c.with_approximation("operator", operator_label(op))
    }
}

/// Canonical RI-MP2 (`rimp2`, closed or open shell) run with `cfg`.
pub fn rimp2_certificate(
    reference: &Certificate,
    aux: &str,
    cfg: &RiMp2Config,
    op: &Operator,
    precision: &PrecisionRecord,
) -> Certificate {
    let mut c = ri_correlation_certificate("rimp2", reference, aux, cfg.frozen_core, op)
        .with_precision(precision);
    if let Some(m) = cfg.metric_op.as_ref() {
        c = c.with_approximation("ri-metric", operator_label(m));
    }
    if let Some(t) = cfg.eri3_screen_thresh {
        c = c.with_approximation("eri3-screen", format!("threshold {t:e}"));
    }
    if let Some(k) = cfg.kappa {
        c = c.with_approximation("kappa", format!("kappa-regularized denominators, {k}"));
    }
    c
}

/// Local RI-MP2 (`[local] scheme = "amplitude-threshold"`): the model
/// certificate ([`Lmp2Certificate::certificate`]) on `reference`.
pub fn local_rimp2_certificate(lmp2: &Lmp2Certificate, reference: &Certificate) -> Certificate {
    lmp2.certificate().with_reference(reference)
}

/// Orbital-optimized RI-MP2 (`oo-rimp2`). A run whose orbital optimization
/// did not converge is capped to [`Grade::Ungraded`].
pub fn oo_rimp2_certificate(
    reference: &Certificate,
    aux: &str,
    frozen_core: usize,
    op: &Operator,
    converged: bool,
    iterations: usize,
    grad_norm: f64,
) -> Certificate {
    let c = ri_correlation_certificate("oo-rimp2", reference, aux, frozen_core, op);
    if converged {
        c
    } else {
        c.cap_grade(
            Grade::Ungraded,
            format!(
                "orbital optimization did not converge ({iterations} iterations, \
                 gradient norm {grad_norm:.2e})"
            ),
        )
    }
}

/// Attenuated RI-MP2 (`att-rimp2`) with the erfc operator of `cfg`.
pub fn att_rimp2_certificate(
    reference: &Certificate,
    aux: &str,
    cfg: &AttenuatedMp2Config,
) -> Certificate {
    let op = Operator::erfc(cfg.omega);
    let c = ri_correlation_certificate("att-rimp2", reference, aux, cfg.frozen_core, &op);
    if cfg.scaling == 1.0 {
        c
    } else {
        c.with_approximation("scaling", format!("correlation scaled by {}", cfg.scaling))
    }
}

/// SCS-MP2 (`scs-mp2`) with the spin-component scales of `cfg`.
pub fn scs_mp2_certificate(reference: &Certificate, aux: &str, cfg: &ScsMp2Config) -> Certificate {
    ri_correlation_certificate(
        "scs-mp2",
        reference,
        aux,
        cfg.frozen_core,
        &Operator::coulomb(),
    )
    .with_approximation("scs", format!("c_os = {}, c_ss = {}", cfg.c_os, cfg.c_ss))
}

/// SCS-MP2 with two terfc operators (`scs-mp2-2terfc`).
pub fn scs_mp2_2terfc_certificate(
    reference: &Certificate,
    aux: &str,
    cfg: &ScsMp2TerfcConfig,
) -> Certificate {
    ri_correlation_certificate(
        "scs-mp2-2terfc",
        reference,
        aux,
        cfg.frozen_core,
        &Operator::coulomb(),
    )
    .with_approximation("scs", format!("c_os = {}, c_ss = {}", cfg.c_os, cfg.c_ss))
    .with_approximation(
        "operator",
        format!(
            "terfc pair, r0 = {} / {} Bohr (bonded / nonbonded)",
            cfg.r0_bonded, cfg.r0_nonbonded
        ),
    )
}

/// Laplace-transformed RI-MP2 (`laplace-mp2`) on `n_quad` points.
pub fn laplace_mp2_certificate(
    reference: &Certificate,
    aux: &str,
    frozen_core: usize,
    op: &Operator,
    n_quad: usize,
) -> Certificate {
    ri_correlation_certificate("laplace-mp2", reference, aux, frozen_core, op)
        .with_approximation("laplace", format!("{n_quad}-point minimax quadrature"))
}

/// Laplace SOS-MP2 (`laplace-sos-mp2`) with `cfg` and `formulation`.
pub fn sos_mp2_certificate(
    reference: &Certificate,
    aux: &str,
    cfg: &SosMp2Config,
    formulation: SosFormulation,
    op: &Operator,
) -> Certificate {
    let c = ri_correlation_certificate("laplace-sos-mp2", reference, aux, cfg.frozen_core, op)
        .with_approximation("sos", format!("c_os = {}, c_ss = 0", cfg.c_os))
        .with_approximation(
            "laplace",
            format!("{}-point minimax quadrature", cfg.n_quad),
        );
    match formulation {
        SosFormulation::Mo | SosFormulation::Ao => c,
        SosFormulation::AoSparse(r) => c.with_approximation(
            "ao-domain",
            format!("AO pseudo-densities restricted to {r} Bohr Boys domains"),
        ),
        SosFormulation::Local(r) => c.with_approximation(
            "local-domain",
            format!("Boys LMO x VV-HV LVO with {r} Bohr virtual domains"),
        ),
    }
}

/// Record an amplitude-threshold `[local]` scheme at `eps` with the kept
/// amplitude fraction. A finite `eps` is capped to [`Grade::Ungraded`]: the
/// validation rows anchor the local schemes by their `eps = 0` limit only.
pub fn with_amplitude_threshold(c: Certificate, eps: f64, keep_fraction: f64) -> Certificate {
    let c = c.with_approximation(
        "local",
        format!("amplitude-threshold eps = {eps:e}, keep_fraction = {keep_fraction}"),
    );
    if eps == 0.0 {
        c
    } else {
        c.cap_grade(
            Grade::Ungraded,
            format!(
                "finite eps = {eps:e}: the local scheme is anchored at eps = 0 only; its \
                 finite-eps error is a measured map, not validated"
            ),
        )
    }
}

/// dRPA by the drCCD Riccati solve (`drpa`). `local` is `(eps, keep_fraction)`
/// under `[local]`, `None` for the exact method. A fixed point that did not
/// converge is capped to [`Grade::Ungraded`].
#[allow(clippy::too_many_arguments)]
pub fn drpa_certificate(
    reference: &Certificate,
    aux: &str,
    frozen_core: usize,
    op: &Operator,
    local: Option<(f64, f64)>,
    converged: bool,
    iterations: usize,
    relres: f64,
) -> Certificate {
    let mut c = ri_correlation_certificate("drpa", reference, aux, frozen_core, op);
    if let Some((eps, keep)) = local {
        c = with_amplitude_threshold(c, eps, keep);
    }
    if converged {
        c
    } else {
        c.cap_grade(
            Grade::Ungraded,
            format!(
                "drCCD fixed point did not converge (relres {relres:.2e} after {iterations} \
                 iterations)"
            ),
        )
    }
}

/// Attenuated MP2 + VV10 (`mp2-v`) run with `cfg`.
pub fn mp2_v_certificate(
    reference: &Certificate,
    aux: &str,
    cfg: &crate::att_vv10::AttVv10Config,
) -> Certificate {
    let omega = cfg
        .omega
        .map_or(String::new(), |w| format!(", omega = {w} Bohr^-1"));
    ri_correlation_certificate(
        "mp2-v",
        reference,
        aux,
        cfg.frozen_core,
        &Operator::coulomb(),
    )
    .with_approximation(
        "operator",
        format!("{:?}, r0 = {} Bohr{omega}", cfg.attenuator, cfg.r0_bohr).to_lowercase(),
    )
    .with_approximation(
        "vv10",
        format!(
            "{:?} damping, NLC grid ({}, {})",
            cfg.vv10_damping, cfg.nlc_grid.n_radial, cfg.nlc_grid.n_angular
        ),
    )
}

/// MP3 (`mp3`).
pub fn mp3_certificate(
    reference: &Certificate,
    aux: &str,
    frozen_core: usize,
    op: &Operator,
) -> Certificate {
    ri_correlation_certificate("mp3", reference, aux, frozen_core, op)
}

/// MP2-based double hybrid (`kind` is `b2plyp` or `dsd-pbep86`) run with
/// `cfg`: scaled RI-MP2 on the functional's own Kohn-Sham `reference`.
pub fn mp2_double_hybrid_certificate(
    kind: &str,
    reference: &Certificate,
    aux: &str,
    cfg: &Mp2DoubleHybridConfig,
) -> Certificate {
    Certificate::for_post_scf(kind, reference)
        .with_ri(aux)
        .with_frozen_core(cfg.frozen_core)
        .with_approximation(
            "spin-component-scaling",
            format!("c_os = {}, c_ss = {}", cfg.c_os, cfg.c_ss),
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::certificate::{BuildStamp, GRADE_SOURCE_PAGE};

    fn rhf() -> Certificate {
        Certificate::for_method_kind("rhf")
            .with_build(BuildStamp::new("v", "c", Some(false)))
            .with_approximation("ri-j", "aux def2-universal-jkfit")
    }

    fn names(c: &Certificate) -> Vec<&str> {
        c.approximations.iter().map(|a| a.name.as_str()).collect()
    }

    #[test]
    fn rimp2_carries_grade_aux_and_settings() {
        let cfg = RiMp2Config {
            frozen_core: 1,
            kappa: Some(1.1),
            ..Default::default()
        };
        let c = rimp2_certificate(
            &rhf(),
            "cc-pvdz-ri",
            &cfg,
            &Operator::coulomb(),
            &PrecisionRecord::f64(),
        );
        assert_eq!(c.capability, "rimp2");
        assert_eq!(c.grade, Grade::Proven);
        assert_eq!(c.grade_source.as_deref(), Some(GRADE_SOURCE_PAGE));
        assert_eq!(names(&c), ["scf ri-j", "ri", "frozen-core", "kappa"]);
        assert_eq!(c.approximations[1].detail, "aux cc-pvdz-ri");
        assert_eq!(c.build.as_ref().map(|b| b.commit.as_str()), Some("c"));
        assert!(c.precision.is_some());
        assert!(c.warnings.is_empty(), "{:?}", c.warnings);
        // A non-Coulomb correlation operator is named.
        let e = rimp2_certificate(
            &rhf(),
            "x",
            &RiMp2Config::default(),
            &Operator::erfc(0.5),
            &PrecisionRecord::f64(),
        );
        assert_eq!(names(&e), ["scf ri-j", "ri", "operator"]);
        assert_eq!(e.approximations[2].detail, "erfc(omega=0.5)");
    }

    #[test]
    fn family_grades_come_from_the_table() {
        let r = rhf();
        let op = Operator::coulomb();
        let cases = [
            (mp3_certificate(&r, "a", 0, &op), "mp3", Grade::Proven),
            (
                oo_rimp2_certificate(&r, "a", 0, &op, true, 5, 1e-7),
                "oo-rimp2",
                Grade::ProvenNarrow,
            ),
            (
                att_rimp2_certificate(&r, "a", &AttenuatedMp2Config::default()),
                "att-rimp2",
                Grade::Proven,
            ),
            (
                scs_mp2_certificate(&r, "a", &ScsMp2Config::default()),
                "scs-mp2",
                Grade::Proven,
            ),
            (
                laplace_mp2_certificate(&r, "a", 0, &op, 7),
                "laplace-mp2",
                Grade::Proven,
            ),
            (
                sos_mp2_certificate(&r, "a", &SosMp2Config::default(), SosFormulation::Mo, &op),
                "laplace-sos-mp2",
                Grade::Ungraded,
            ),
            (
                mp2_v_certificate(&r, "a", &crate::att_vv10::AttVv10Config::default()),
                "mp2-v",
                Grade::ProvenNarrow,
            ),
        ];
        for (c, kind, g) in cases {
            assert_eq!(c.capability, kind);
            assert_eq!(c.grade, g, "{kind}");
            assert_eq!(
                c.grade,
                ferric_core::certificate::method_kind_grade(kind),
                "{kind}"
            );
        }
    }

    #[test]
    fn method_settings_are_recorded() {
        let r = rhf();
        let op = Operator::coulomb();
        let l = laplace_mp2_certificate(&r, "a", 0, &op, 5);
        assert_eq!(l.approximations[2].detail, "5-point minimax quadrature");
        let s = sos_mp2_certificate(
            &r,
            "a",
            &SosMp2Config::default(),
            SosFormulation::AoSparse(4.0),
            &op,
        );
        assert_eq!(names(&s), ["scf ri-j", "ri", "sos", "laplace", "ao-domain"]);
        let a = att_rimp2_certificate(
            &r,
            "a",
            &AttenuatedMp2Config {
                omega: 0.25,
                ..Default::default()
            },
        );
        assert_eq!(a.approximations[2].detail, "erfc(omega=0.25)");
        let scs = scs_mp2_certificate(
            &r,
            "a",
            &ScsMp2Config {
                c_os: 1.2,
                c_ss: 0.5,
                ..Default::default()
            },
        );
        assert_eq!(scs.approximations[2].detail, "c_os = 1.2, c_ss = 0.5");
    }

    #[test]
    fn drpa_exact_local_and_unconverged() {
        let op = Operator::coulomb();
        let exact = drpa_certificate(&rhf(), "a", 0, &op, None, true, 9, 1e-9);
        assert_eq!(exact.capability, "drpa");
        assert_eq!(exact.grade, Grade::ProvenNarrow);
        assert!(!names(&exact).contains(&"local"));
        let anchor = drpa_certificate(&rhf(), "a", 0, &op, Some((0.0, 1.0)), true, 9, 1e-9);
        assert_eq!(anchor.grade, Grade::ProvenNarrow, "eps = 0 is the anchor");
        assert_eq!(
            anchor.approximations[2].detail,
            "amplitude-threshold eps = 0e0, keep_fraction = 1"
        );
        let finite = drpa_certificate(&rhf(), "a", 0, &op, Some((1e-4, 0.3)), true, 9, 1e-9);
        assert_eq!(finite.grade, Grade::Ungraded);
        assert!(finite.warnings[0].contains("finite eps = 1e-4"));
        let bad = drpa_certificate(&rhf(), "a", 0, &op, None, false, 40, 1e-3);
        assert_eq!(bad.grade, Grade::Ungraded);
        assert!(bad.warnings[0].contains("did not converge (relres 1.00e-3 after 40"));
    }

    #[test]
    fn unconverged_orbital_optimization_or_reference_is_ungraded() {
        let op = Operator::coulomb();
        let c = oo_rimp2_certificate(&rhf(), "a", 0, &op, false, 50, 3e-3);
        assert_eq!(c.grade, Grade::Ungraded);
        assert!(
            c.warnings[0].contains("orbital optimization did not converge (50 iterations"),
            "{:?}",
            c.warnings
        );
        let bad = rhf().cap_grade(Grade::Ungraded, "SCF did not converge");
        let m = mp3_certificate(&bad, "a", 0, &op);
        assert_eq!(m.grade, Grade::Ungraded);
    }

    #[test]
    fn double_hybrids_are_spike_with_their_scaling() {
        use crate::double_hybrid::DoubleHybridKind;
        let ks = Certificate::for_method_kind("ksdft")
            .with_build(BuildStamp::new("v", "c", Some(false)))
            .with_approximation("xc-grid", "(75, 302)");
        for (kind, dk) in [
            ("b2plyp", DoubleHybridKind::B2plyp),
            ("dsd-pbep86", DoubleHybridKind::DsdPbep86),
        ] {
            let mut cfg = dk.mp2_config();
            cfg.frozen_core = 1;
            let c = mp2_double_hybrid_certificate(kind, &ks, "cc-pvdz-ri", &cfg);
            assert_eq!(c.capability, kind);
            assert_eq!(c.grade, Grade::Spike, "{kind}");
            assert_eq!(c.grade, ferric_core::certificate::method_kind_grade(kind));
            assert_eq!(
                names(&c),
                ["scf xc-grid", "ri", "frozen-core", "spin-component-scaling"]
            );
            assert_eq!(
                c.approximations[3].detail,
                format!("c_os = {}, c_ss = {}", cfg.c_os, cfg.c_ss)
            );
        }
        // An unconverged reference caps the grade below Spike.
        let bad = ks.cap_grade(Grade::Ungraded, "SCF did not converge");
        let c = mp2_double_hybrid_certificate(
            "b2plyp",
            &bad,
            "a",
            &crate::double_hybrid::DoubleHybridKind::B2plyp.mp2_config(),
        );
        assert_eq!(c.grade, Grade::Ungraded);
    }
}
