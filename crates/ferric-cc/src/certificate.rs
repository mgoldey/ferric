//! [`Certificate`](ferric_core::certificate::Certificate)s of the coupled-cluster results (`ccsd`, `ccd`,
//! `ccsd(t)`, `linlccd`): capability and grade from
//! `ferric_core::certificate::METHOD_KIND_GRADES`, folded with the SCF
//! reference ([`Certificate::for_post_scf`](ferric_core::certificate::Certificate::for_post_scf)), plus the settings of the run.
//! The build stamp comes from the reference.

use ferric_core::certificate::Certificate;
use ferric_mp2::certificate::with_amplitude_threshold;

use crate::ccsd_closed_shell::LadderRoute;
use crate::linlccd::LadderVariant;
use crate::{CcConfig, CcResult};

/// CCSD / CCD / CCSD(T) (`kind`) run with `cfg`. `r` names the VVVV ladder
/// route the spin-adapted solver took (the spin-orbital solvers record none).
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
}
