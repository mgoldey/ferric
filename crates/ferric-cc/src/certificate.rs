//! [`Certificate`](ferric_core::certificate::Certificate)s of the coupled-cluster results (`ccsd`, `ccd`,
//! `ccsd(t)`, `linlccd`): capability and grade from
//! `ferric_core::certificate::METHOD_KIND_GRADES`, folded with the SCF
//! reference ([`Certificate::for_post_scf`](ferric_core::certificate::Certificate::for_post_scf)), plus the settings of the run.
//! The build stamp comes from the reference.

use ferric_core::certificate::Certificate;
use ferric_mp2::certificate::{with_amplitude_threshold, with_anchor_in_scope};

use crate::ccsd_closed_shell::LadderRoute;
use crate::double_hybrid::DoubleHybridConfig;
use crate::linlccd::LadderVariant;
use crate::{CcConfig, CcResult};

/// Whether the run is what `validation_cc.rs` runs: `ccsd` on RHF with every
/// electron correlated.
fn in_ccsd_anchor_scope(kind: &str, reference: &Certificate, cfg: &CcConfig) -> bool {
    kind == "ccsd" && reference.capability == "rhf" && cfg.frozen_core == 0
}

/// CCSD / CCD / CCSD(T) (`kind`) run with `cfg`. `r` names the VVVV ladder
/// route the spin-adapted solver took (the spin-orbital solvers record none).
/// `ccsd` on an RHF `reference` with every electron correlated carries the
/// CCSD energy anchor: `validation_cc.rs` runs both drivers and both ladder
/// routes there (its frozen-core run is one system, which a certificate
/// cannot tell apart). `ccd` and `ccsd(t)` carry none.
pub fn cc_certificate(
    kind: &str,
    reference: &Certificate,
    aux: &str,
    cfg: &CcConfig,
    r: &CcResult,
) -> Certificate {
    let c = Certificate::for_post_scf(kind, reference)
        .with_ri(aux)
        .with_frozen_core(cfg.frozen_core);
    let c = with_anchor_in_scope(c, "CCSD", in_ccsd_anchor_scope(kind, reference, cfg));
    match r.ladder_route {
        Some(LadderRoute::Ri { block }) => c.with_approximation(
            "ri-ladder",
            format!("integral-direct VVVV ladder, block {block}"),
        ),
        Some(LadderRoute::Dense) | None => c,
    }
}

/// LinLCCD (`linlccd`) with `variant`; `local` is `(eps, keep_fraction)`
/// under `[local]`, `None` for the exact method. A local PCG that did not
/// converge is a warning (the run still reports its energy).
pub fn linlccd_certificate(
    reference: &Certificate,
    aux: &str,
    frozen_core: usize,
    variant: LadderVariant,
    local: Option<(f64, f64)>,
    cg_converged: bool,
) -> Certificate {
    let mut c = Certificate::for_post_scf("linlccd", reference)
        .with_ri(aux)
        .with_frozen_core(frozen_core)
        .with_approximation("variant", variant.as_str());
    if let Some((eps, keep)) = local {
        c = with_amplitude_threshold(c, eps, keep);
    }
    if !cg_converged {
        c.push_warning("local LinLCCD PCG did not converge");
    }
    c
}

/// ωB97X-L-V (`wb97x-l-v`) run with `cfg`: λ-scaled short-range LinLCCD on
/// the functional's own Kohn-Sham `reference`. The long-range correlation is
/// VV10 inside the functional, so the amplitudes see only erfc(ωr)/r.
pub fn wb97x_l_v_certificate(
    reference: &Certificate,
    aux: &str,
    cfg: &DoubleHybridConfig,
) -> Certificate {
    Certificate::for_post_scf("wb97x-l-v", reference)
        .with_ri(aux)
        .with_frozen_core(cfg.cc.frozen_core)
        .with_approximation(
            "double-hybrid",
            format!(
                "lambda = {}, omega = {} Bohr^-1, erfc-attenuated LinLCCD",
                cfg.lambda, cfg.omega
            ),
        )
        .with_approximation("variant", cfg.variant.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::certificate::Grade;
    use ndarray::Array4;

    fn cc(route: Option<LadderRoute>) -> CcResult {
        CcResult {
            correlation_energy: -0.2,
            t1: None,
            t2: Array4::zeros((0, 0, 0, 0)),
            ladder_route: route,
        }
    }

    #[test]
    fn cc_kinds_carry_their_table_grades() {
        let rhf = Certificate::for_method_kind("rhf");
        let cfg = CcConfig {
            frozen_core: 1,
            ..Default::default()
        };
        for (kind, g) in [
            ("ccsd", Grade::Proven),
            ("ccd", Grade::ProvenNarrow),
            ("ccsd(t)", Grade::ProvenNarrow),
        ] {
            let c = cc_certificate(kind, &rhf, "cc-pvdz-ri", &cfg, &cc(None));
            assert_eq!(c.capability, kind);
            assert_eq!(c.grade, g, "{kind}");
            let names: Vec<&str> = c.approximations.iter().map(|a| a.name.as_str()).collect();
            assert_eq!(names, ["ri", "frozen-core"]);
        }
        let ri = cc_certificate(
            "ccsd",
            &rhf,
            "a",
            &CcConfig::default(),
            &cc(Some(LadderRoute::Ri { block: 8 })),
        );
        assert_eq!(
            ri.approximations[1].detail,
            "integral-direct VVVV ladder, block 8"
        );
    }

    /// Only `ccsd`, on RHF, all-electron, carries the anchor.
    #[test]
    fn ccsd_anchor_is_attached_inside_the_evidence_scope_only() {
        let rhf = Certificate::for_method_kind("rhf");
        let run = |kind: &str, reference: &Certificate, frozen_core: usize| {
            let cfg = CcConfig {
                frozen_core,
                ..Default::default()
            };
            cc_certificate(kind, reference, "cc-pvdz-ri", &cfg, &cc(None))
        };
        let inside = run("ccsd", &rhf, 0);
        let m = inside.measured_error.as_ref().expect("anchor attached");
        assert_eq!(
            *m,
            ferric_core::certificate::energy_anchor("CCSD")
                .unwrap()
                .measured_error()
        );
        assert_eq!(m.evidence, "crates/ferric-cc/tests/validation_cc.rs");
        let outside = [
            ("frozen core", run("ccsd", &rhf, 1)),
            ("ccd", run("ccd", &rhf, 0)),
            ("ccsd(t)", run("ccsd(t)", &rhf, 0)),
            (
                "uhf reference",
                run("ccsd", &Certificate::for_method_kind("uhf"), 0),
            ),
            (
                "ks reference",
                run("ccsd", &Certificate::for_method_kind("ksdft"), 0),
            ),
        ];
        for (what, c) in outside {
            assert!(c.measured_error.is_none(), "{what}: anchor attached");
        }
    }

    #[test]
    fn linlccd_exact_and_local() {
        let rhf = Certificate::for_method_kind("rhf");
        let v = LadderVariant::Hh;
        let exact = linlccd_certificate(&rhf, "a", 0, v, None, true);
        assert_eq!(exact.grade, Grade::ProvenNarrow);
        assert_eq!(exact.approximations[1].detail, v.as_str());
        let local = linlccd_certificate(&rhf, "a", 0, v, Some((1e-5, 0.5)), false);
        assert_eq!(local.grade, Grade::Ungraded);
        assert_eq!(local.warnings.len(), 2, "{:?}", local.warnings);
        assert!(local.warnings[1].contains("PCG did not converge"));
    }

    #[test]
    fn wb97x_l_v_is_proven_narrow_with_lambda_and_omega() {
        let ks = Certificate::for_method_kind("ksdft");
        let cfg = DoubleHybridConfig {
            lambda: 0.5,
            omega: 0.2,
            ..Default::default()
        };
        let c = wb97x_l_v_certificate(&ks, "cc-pvdz-ri", &cfg);
        assert_eq!(c.capability, "wb97x-l-v");
        assert_eq!(c.grade, Grade::ProvenNarrow);
        let names: Vec<&str> = c.approximations.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["ri", "double-hybrid", "variant"]);
        assert!(c.approximations[1]
            .detail
            .starts_with("lambda = 0.5, omega = 0.2 Bohr^-1"));
        let bad = ks.cap_grade(Grade::Ungraded, "SCF did not converge");
        assert_eq!(
            wb97x_l_v_certificate(&bad, "a", &cfg).grade,
            Grade::Ungraded
        );
    }
}
