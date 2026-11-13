//! [`Certificate`] of a TDA / Casida TDDFT result: capability and grade from
//! `ferric_core::certificate::METHOD_KIND_GRADES` (`tda`, `tddft`), folded with
//! the SCF reference ([`Certificate::for_post_scf`]), plus the settings of the
//! run and the eigensolver that produced the roots, with their residuals. The
//! build stamp comes from the reference.

use ferric_core::certificate::Certificate;
use ferric_dft::grid::AtomicGridConfig;

use crate::{TddftConfig, TddftMethod, TddftResult, TddftSolver};

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

/// The `eigensolver` approximation of `result`: which solver, how many roots
/// and the largest root residual (and, for Davidson, the tolerance it was
/// converged to and the iteration count).
pub fn eigensolver_label(result: &TddftResult, cfg: &TddftConfig) -> String {
    let max_res = result.residuals.iter().fold(0.0_f64, |m, &r| m.max(r));
    let n = result.residuals.len();
    match result.solver {
        TddftSolver::Davidson => format!(
            "davidson, {n} roots, max residual {max_res:.2e} (conv_tol {:.0e}), {} iterations",
            cfg.davidson.conv_tol, result.iterations
        ),
        _ => format!("dense, {n} roots, max residual {max_res:.2e}"),
    }
}

/// TDA (`tda`) or Casida TDDFT (`tddft`), per `cfg.method`, run with `cfg` on
/// `reference`, giving `result`. `grid_explicit` says whether the caller set
/// the KS grid.
///
/// A KS run records the `response-grid` (the reference SCF's and the f_xc
/// kernel's shared grid) and the exact-exchange fraction; a meta-GGA run also
/// warns that its roots are grid-sensitive. An HF reference (CIS / TDHF) has
/// no grid and no kernel. Every run records the `eigensolver` and its root
/// residuals ([`eigensolver_label`]).
pub fn tddft_certificate(
    reference: &Certificate,
    aux: &str,
    cfg: &TddftConfig,
    result: &TddftResult,
    grid_explicit: bool,
) -> Certificate {
    let kind = match cfg.method {
        TddftMethod::Tda => "tda",
        TddftMethod::Casida => "tddft",
    };
    let mut c = Certificate::for_post_scf(kind, reference)
        .with_ri(aux)
        .with_approximation("eigensolver", eigensolver_label(result, cfg));
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

    fn result(solver: TddftSolver) -> TddftResult {
        TddftResult {
            excitation_energies: vec![0.3, 0.4],
            oscillator_strengths: vec![0.0, 0.1],
            transition_dipoles: vec![[0.0; 3]; 2],
            method: TddftMethod::Tda,
            c_hf: 0.0,
            fxc_included: false,
            solver,
            residuals: vec![2e-7, 5e-7],
            iterations: if solver == TddftSolver::Davidson {
                9
            } else {
                0
            },
        }
    }

    #[test]
    fn eigensolver_and_residuals_are_recorded() {
        let cfg = TddftConfig::new(2, TddftMethod::Tda);
        let rhf = Certificate::for_method_kind("rhf");
        for (solver, want) in [
            (
                TddftSolver::Davidson,
                "davidson, 2 roots, max residual 5.00e-7 (conv_tol 1e-6), 9 iterations",
            ),
            (TddftSolver::Dense, "dense, 2 roots, max residual 5.00e-7"),
        ] {
            let c = tddft_certificate(&rhf, "a", &cfg, &result(solver), false);
            let e = c
                .approximations
                .iter()
                .find(|a| a.name == "eigensolver")
                .expect("eigensolver");
            assert_eq!(e.detail, want);
        }
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
        let rhf_cert = |cfg: &TddftConfig| {
            let r = result(TddftSolver::Dense);
            tddft_certificate(&Certificate::for_method_kind("rhf"), "a", cfg, &r, false)
        };
        let c = rhf_cert(&tda);
        assert_eq!(c.capability, "tda");
        assert_eq!(c.grade, Grade::ProvenNarrow);
        let names: Vec<&str> = c.approximations.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["ri", "eigensolver"], "an HF reference has no grid");
        let cas = TddftConfig::new(3, TddftMethod::Casida);
        assert_eq!(rhf_cert(&cas).capability, "tddft");
        let bad = ks().cap_grade(Grade::Ungraded, "SCF did not converge");
        assert_eq!(
            tddft_certificate(&bad, "a", &tda, &result(TddftSolver::Dense), false).grade,
            Grade::Ungraded
        );
    }

    #[test]
    fn ks_run_records_its_grid_and_the_meta_gga_default() {
        let mut cfg = TddftConfig::new(3, TddftMethod::Tda);
        cfg.xc = Some("PBE".to_string());
        let c = tddft_certificate(&ks(), "a", &cfg, &result(TddftSolver::Dense), false);
        let g = c
            .approximations
            .iter()
            .find(|a| a.name == "response-grid")
            .expect("response-grid");
        assert!(g.detail.ends_with("[default]"), "{}", g.detail);
        assert!(c.warnings.is_empty(), "{:?}", c.warnings);
        let explicit = tddft_certificate(&ks(), "a", &cfg, &result(TddftSolver::Dense), true);
        assert!(grid_detail(&explicit).ends_with("[explicit grid keys]"));

        cfg.xc = Some("r2SCAN".to_string());
        cfg.grid = ferric_dft::grid::response_default_grid(Some("r2SCAN")).unwrap();
        let m = tddft_certificate(&ks(), "a", &cfg, &result(TddftSolver::Dense), false);
        let d = grid_detail(&m);
        assert!(
            d.starts_with("(150, 434)") && d.ends_with("[meta-GGA response default]"),
            "{d}"
        );
        assert_eq!(m.warnings.len(), 1);
        assert!(m.warnings[0].contains("depend on the integration grid"));
    }
}
