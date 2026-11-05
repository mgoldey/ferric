//! [`Certificate`] of a TDA / Casida TDDFT result: capability and grade from
//! `ferric_core::certificate::METHOD_KIND_GRADES` (`tda`, `tddft`), folded with
//! the SCF reference ([`Certificate::for_post_scf`]), plus the settings of the
//! run. The build stamp comes from the reference.

use ferric_core::certificate::Certificate;
use ferric_dft::grid::AtomicGridConfig;

use crate::{TddftConfig, TddftMethod};

/// The grid of a KS TDDFT run as text, with why it was chosen: the reference
/// SCF and the f_xc kernel share it. `explicit` is true when the caller set a
/// grid; with none set, a meta-GGA `xc` gets the finer response default.
pub fn response_grid_label(xc: &str, g: &AtomicGridConfig, explicit: bool) -> String {
    let prune = g
        .prune
        .map_or("flat".to_string(), |p| format!("{p:?}").to_lowercase());
    let cut = g
        .weight_cut
        .map_or("no weight cut".to_string(), |w| format!("weight cut {w:e}"));
    let why = if explicit {
        "explicit grid keys"
    } else if ferric_dft::grid::response_default_grid(Some(xc)).is_some() {
        "meta-GGA response default"
    } else {
        "default"
    };
    format!("({}, {}), {prune}, {cut} [{why}]", g.n_radial, g.n_angular)
}

/// TDA (`tda`) or Casida TDDFT (`tddft`), per `cfg.method`, run with `cfg` on
/// `reference`. `grid_explicit` says whether the caller set the KS grid.
///
/// A KS run records the `response-grid` (the reference SCF's and the f_xc
/// kernel's shared grid) and the exact-exchange fraction; a meta-GGA run also
/// warns that its roots are grid-sensitive. An HF reference (CIS / TDHF) has
/// no grid and no kernel.
pub fn tddft_certificate(
    reference: &Certificate,
    aux: &str,
    cfg: &TddftConfig,
    grid_explicit: bool,
) -> Certificate {
    let kind = match cfg.method {
        TddftMethod::Tda => "tda",
        TddftMethod::Casida => "tddft",
    };
    let mut c = Certificate::for_post_scf(kind, reference).with_ri(aux);
    if let Some(xc) = cfg.xc.as_deref() {
        c = c.with_approximation(
            "response-grid",
            response_grid_label(xc, &cfg.grid, grid_explicit),
        );
        if ferric_dft::grid::response_default_grid(Some(xc)).is_some() {
            c.push_warning(format!(
                "{xc} is a meta-GGA: its TDDFT roots depend on the integration grid. The \
                 validated cases are r2SCAN on water/6-31G on the flat (75, 110) grid; SCAN is \
                 not grid-converged at 1e-4 eV on any grid tested"
            ));
        }
    }
    if let Some(c_hf) = cfg.c_hf_override {
        c = c.with_approximation("c-hf-override", format!("exact-exchange fraction {c_hf}"));
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::certificate::Grade;

    fn ks() -> Certificate {
        Certificate::for_method_kind("ksdft")
    }

    fn grid_detail(c: &Certificate) -> &str {
        &c.approximations
            .iter()
            .find(|a| a.name == "response-grid")
            .expect("response-grid")
            .detail
    }

    #[test]
    fn grade_follows_the_method_and_caps_at_the_reference() {
        let tda = TddftConfig::new(3, TddftMethod::Tda);
        let c = tddft_certificate(&Certificate::for_method_kind("rhf"), "a", &tda, false);
        assert_eq!(c.capability, "tda");
        assert_eq!(c.grade, Grade::ProvenNarrow);
        let names: Vec<&str> = c.approximations.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["ri"], "an HF reference has no grid");
        let cas = TddftConfig::new(3, TddftMethod::Casida);
        assert_eq!(
            tddft_certificate(&Certificate::for_method_kind("rhf"), "a", &cas, false).capability,
            "tddft"
        );
        let bad = ks().cap_grade(Grade::Ungraded, "SCF did not converge");
        assert_eq!(
            tddft_certificate(&bad, "a", &tda, false).grade,
            Grade::Ungraded
        );
    }

    #[test]
    fn ks_run_records_its_grid_and_the_meta_gga_default() {
        let mut cfg = TddftConfig::new(3, TddftMethod::Tda);
        cfg.xc = Some("PBE".to_string());
        let c = tddft_certificate(&ks(), "a", &cfg, false);
        let g = c
            .approximations
            .iter()
            .find(|a| a.name == "response-grid")
            .expect("response-grid");
        assert!(g.detail.ends_with("[default]"), "{}", g.detail);
        assert!(c.warnings.is_empty(), "{:?}", c.warnings);
        let explicit = tddft_certificate(&ks(), "a", &cfg, true);
        assert!(grid_detail(&explicit).ends_with("[explicit grid keys]"));

        cfg.xc = Some("r2SCAN".to_string());
        cfg.grid = ferric_dft::grid::response_default_grid(Some("r2SCAN")).unwrap();
        let m = tddft_certificate(&ks(), "a", &cfg, false);
        let d = grid_detail(&m);
        assert!(
            d.starts_with("(150, 434)") && d.ends_with("[meta-GGA response default]"),
            "{d}"
        );
        assert_eq!(m.warnings.len(), 1);
        assert!(m.warnings[0].contains("depend on the integration grid"));
    }
}
