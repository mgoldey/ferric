//! [`Certificate`] of a TDA / Casida TDDFT result: capability and grade from
//! `ferric_core::certificate::METHOD_KIND_GRADES` (`tda`, `tddft`), folded with
//! the SCF reference ([`Certificate::for_post_scf`]), plus the settings of the
//! run and the eigensolver that produced the roots, with their residuals. The
//! build stamp comes from the reference.
//!
//! The Proven (narrow) grade holds where a PySCF reference exists: singlets
//! of any accepted functional that is not range-separated, range-separated
//! singlets of the functionals in [`VALIDATED_RSH`], triplets of the
//! references in [`VALIDATED_TRIPLET`], and spin-conserving roots on an
//! unrestricted (UHF / UKS) reference of the references in
//! [`VALIDATED_UNRESTRICTED`]. Any other combination is capped at
//! Smoke with the reason. A functional enters an envelope by name, with its
//! reference data.

use ferric_core::certificate::{Certificate, Grade};
use ferric_dft::grid::AtomicGridConfig;

use crate::{ResponseSpin, TddftConfig, TddftMethod, TddftResult, TddftSolver};

/// Range-separated hybrids whose singlet TDA and Casida roots have a PySCF
/// reference (`tests/validation_tddft_rsh.rs`), compared by [`functional_key`].
pub const VALIDATED_RSH: &[&str] = &["HYB_GGA_XC_CAM_B3LYP", "HYB_GGA_XC_WB97X"];

/// References (`"HF"` for none) whose triplet TDA and Casida roots have a
/// PySCF reference (`tests/validation_tddft_rsh.rs`), compared by
/// [`functional_key`].
pub const VALIDATED_TRIPLET: &[&str] = &[
    "HF",
    "LDA",
    "PBE",
    "B3LYP",
    "r2SCAN",
    "HYB_GGA_XC_CAM_B3LYP",
    "HYB_GGA_XC_WB97X",
];

/// References (`"HF"` for none) whose spin-conserving TDA and Casida roots
/// on an unrestricted (UHF / UKS) reference have a PySCF reference
/// (`tests/validation_tddft_open_shell.rs`), compared by [`functional_key`].
pub const VALIDATED_UNRESTRICTED: &[&str] = &[
    "HF",
    "LDA",
    "PBE",
    "B3LYP",
    "r2SCAN",
    "TPSSh",
    "HYB_GGA_XC_CAM_B3LYP",
];

/// A functional name as the envelope compares it: upper case, without `-`
/// and `_` (the friendly-name normalization of `ferric_dft::libxc`).
pub fn functional_key(name: &str) -> String {
    name.to_uppercase()
        .chars()
        .filter(|c| *c != '-' && *c != '_')
        .collect()
}

/// Why `(xc, spin)` (with long-range ω of the resolved exchange) is outside
/// the validated envelope, or `None` inside it.
fn outside_envelope(xc: Option<&str>, spin: Option<ResponseSpin>, omega: f64) -> Option<String> {
    let name = xc.unwrap_or("HF");
    let key = functional_key(name);
    let listed = |list: &[&str]| list.iter().any(|f| functional_key(f) == key);
    match spin {
        None if !listed(VALIDATED_UNRESTRICTED) => Some(format!(
            "unrestricted response on {name} has no PySCF reference (validated: {})",
            VALIDATED_UNRESTRICTED.join(", ")
        )),
        None => None,
        Some(ResponseSpin::Triplet) if !listed(VALIDATED_TRIPLET) => Some(format!(
            "triplet response on {name} has no PySCF reference (validated: {})",
            VALIDATED_TRIPLET.join(", ")
        )),
        Some(ResponseSpin::Singlet) if omega != 0.0 && !listed(VALIDATED_RSH) => Some(format!(
            "range-separated response on {name} has no PySCF reference (validated: {})",
            VALIDATED_RSH.join(", ")
        )),
        _ => None,
    }
}

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
/// residuals ([`eigensolver_label`]); a triplet run records `spin`, a
/// range-separated one its `long-range-exchange`. A combination outside the
/// validated envelope (module docs) is capped at Smoke.
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
    c = with_exchange_and_spin(c, aux, cfg, result);
    if let Some(why) = outside_envelope(cfg.xc.as_deref(), result.spin, result.omega) {
        c = c.cap_grade(Grade::Smoke, why);
    }
    c
}

/// The model choices of [`tddft_certificate`] beyond the reference: a
/// `c-hf-override`, the `long-range-exchange` of a range-separated hybrid and
/// the `spin` of a triplet run.
fn with_exchange_and_spin(
    mut c: Certificate,
    aux: &str,
    cfg: &TddftConfig,
    result: &TddftResult,
) -> Certificate {
    if let Some(c_hf) = cfg.c_hf_override {
        c = c.with_approximation("c-hf-override", format!("exact-exchange fraction {c_hf}"));
    }
    if result.omega != 0.0 {
        c = c.with_approximation(
            "long-range-exchange",
            format!(
                "c_sr {} K[1/r] + (c_lr {} - c_sr) K[erf({} r)/r]; erf integrals fitted in \
                 the erf metric over {aux}",
                result.c_hf, result.c_lr, result.omega
            ),
        );
    }
    match result.spin {
        Some(ResponseSpin::Triplet) => {
            c = c.with_approximation(
                "spin",
                "M_S = 0 triplet, f_aa - f_ab kernel, no Coulomb term",
            );
        }
        None => {
            let s2 = result.s2.iter().fold(f64::NAN, |m: f64, &x| m.max(x));
            c = c.with_approximation(
                "spin",
                format!(
                    "unrestricted reference (<S^2> = {:.4}), spin-conserving excitations; \
                     roots are not spin-adapted (largest root <S^2> {s2:.4}{})",
                    result.s2_reference,
                    if cfg.method == TddftMethod::Casida {
                        ", from the renormalized X part"
                    } else {
                        ""
                    }
                ),
            );
        }
        Some(ResponseSpin::Singlet) => {}
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

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
            c_lr: 0.0,
            omega: 0.0,
            spin: Some(ResponseSpin::Singlet),
            reference: ferric_scf::Spin::Restricted,
            s2: vec![0.0; 2],
            s2_reference: 0.0,
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

    /// Validated combinations keep Proven (narrow); a triplet or a
    /// range-separated singlet without a reference is capped at Smoke, by
    /// name.
    #[test]
    fn envelope_caps_unvalidated_combinations_by_name() {
        let rhf = Certificate::for_method_kind("rhf");
        let grade = |xc: Option<&str>, spin, omega| {
            let mut cfg = TddftConfig::new(3, TddftMethod::Tda);
            cfg.xc = xc.map(str::to_string);
            let mut r = result(TddftSolver::Dense);
            r.spin = Some(spin);
            r.omega = omega;
            let base = if xc.is_some() { ks() } else { rhf.clone() };
            tddft_certificate(&base, "a", &cfg, &r, false).grade
        };
        use ResponseSpin::{Singlet, Triplet};
        assert_eq!(grade(Some("PBE"), Singlet, 0.0), Grade::ProvenNarrow);
        assert_eq!(grade(Some("cam-b3lyp"), Singlet, 0.33), Grade::Smoke);
        assert_eq!(
            grade(Some("HYB_GGA_XC_CAM_B3LYP"), Singlet, 0.33),
            Grade::ProvenNarrow
        );
        assert_eq!(grade(Some("HYB_GGA_XC_HSE06"), Singlet, 0.11), Grade::Smoke);
        assert_eq!(grade(None, Triplet, 0.0), Grade::ProvenNarrow);
        assert_eq!(
            grade(Some("hyb-gga-xc-wb97x"), Triplet, 0.3),
            Grade::ProvenNarrow
        );
        assert_eq!(grade(Some("BLYP"), Triplet, 0.0), Grade::Smoke);
    }

    /// An unrestricted run is graded against [`VALIDATED_UNRESTRICTED`], not
    /// the restricted lists: a functional validated for restricted singlets
    /// only (BLYP) is capped at Smoke, a listed one keeps Proven (narrow), and
    /// the certificate records the reference and root `<S^2>`.
    ///
    /// Mutation (applied and reverted, 2026-11-14): the unrestricted
    /// envelope check disabled (`None if false && ...`) leaves BLYP at Proven
    /// (narrow) and fails the Smoke assertion.
    #[test]
    fn unrestricted_runs_use_their_own_envelope_and_record_s2() {
        let run = |xc: Option<&str>, omega| {
            let mut cfg = TddftConfig::new(3, TddftMethod::Tda);
            cfg.xc = xc.map(str::to_string);
            let mut r = result(TddftSolver::Dense);
            r.spin = None;
            r.reference = ferric_scf::Spin::Unrestricted;
            r.s2 = vec![0.76, 1.83];
            r.s2_reference = 0.7553;
            r.omega = omega;
            let base = if xc.is_some() {
                ks()
            } else {
                Certificate::for_method_kind("uhf")
            };
            tddft_certificate(&base, "a", &cfg, &r, false)
        };
        assert_eq!(run(None, 0.0).grade, Grade::ProvenNarrow);
        assert_eq!(run(Some("tpssh"), 0.0).grade, Grade::ProvenNarrow);
        assert_eq!(
            run(Some("HYB_GGA_XC_CAM_B3LYP"), 0.33).grade,
            Grade::ProvenNarrow
        );
        assert_eq!(run(Some("BLYP"), 0.0).grade, Grade::Smoke);
        assert_eq!(run(Some("HYB_GGA_XC_WB97X"), 0.3).grade, Grade::Smoke);
        let c = run(Some("PBE"), 0.0);
        let spin = c
            .approximations
            .iter()
            .find(|a| a.name == "spin")
            .expect("spin approximation");
        assert!(
            spin.detail.contains("<S^2> = 0.7553") && spin.detail.contains("1.8300"),
            "{}",
            spin.detail
        );
    }
}
