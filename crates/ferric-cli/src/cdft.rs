//! `[cdft]`: constrained DFT (Wu-Van Voorhis) from TOML, the CLI form of
//! Python `run_cdft`.
//!
//! Present, the section turns `method.kind = "uhf"` into the constrained
//! solve of [`ferric_scf::cdft_driver::solve_cdft_uhf`]: UHF, or UKS with
//! `[dft] functional`. The constraint operator is the Becke fuzzy-cell weight
//! of each fragment, `target` is a fragment POPULATION in electrons (not a net
//! charge), and the multipliers are found by a nested lambda-Newton loop around
//! the ordinary open-shell SCF.
//!
//! Python keyword to TOML key:
//!
//! | `run_cdft` | TOML |
//! |---|---|
//! | `CdftConstraint(atoms, target, kind)` | one `[[cdft.constraint]]` with `atoms`, `target`, `kind` |
//! | `lambda_tol`, `max_outer`, `stability_descent` | `[cdft]` keys of the same names |
//! | `grid_radial`, `grid_angular` | `[cdft]` keys of the same names |
//! | `functional` | `[dft] functional` |
//! | `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`, ... | `[scf] max_iter`, `energy_conv`, `density_conv`, ... |
//!
//! The lambda loop has its own iteration cap (`[cdft] max_outer`); `[scf]
//! max_iter` is the cap of each inner SCF. The grid of the weight operator
//! (and of the XC functional, when there is one) is set in `[cdft]`, so the
//! `[dft]` grid keys are refused here: two places to size one grid would
//! disagree.
//!
//! Every section and every `[scf]` key outside what the constrained solve
//! reads is refused at load time ([`check_sections`]), the convention of
//! `[cas]`. The Wu-Van Voorhis coupling of two diabats (`cdft_coupling`) needs
//! two solves sharing one Hamiltonian and is Python-only.

use crate::config::Config;
use crate::error::CliError;
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_dft::cdft::{Constraint, SpinChannel};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::PreparedSystem;
use serde::Deserialize;

/// `[cdft]`: the constraints and the lambda loop of a constrained run.
#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct CdftCfg {
    /// The constraints, one `[[cdft.constraint]]` table each; at least one.
    #[serde(default)]
    pub constraint: Vec<CdftConstraintCfg>,
    /// The solve stops when `max |N_C - target_C|` is below this, in
    /// electrons. Default `1e-5`. A flat stretch of `N(lambda)` costs outer
    /// iterations, not accuracy; a looser tolerance accepts a lambda further
    /// from the root.
    pub lambda_tol: Option<f64>,
    /// Cap on the outer lambda-Newton iterations. Default `30`. Exhausting it
    /// is an error; nothing unconverged is reported.
    pub max_outer: Option<usize>,
    /// After the outer loop converges, check the lambda-augmented orbital
    /// Hessian and descend if the constrained solution is a saddle (UHF only;
    /// skipped on a KS reference). Default `true`, unlike `[scf]
    /// stability_descent` (the unconstrained descent), which is refused here.
    pub stability_descent: Option<bool>,
    /// Radial points of the Becke/Lebedev grid of the weight operator, and of
    /// the XC functional when one is set. With both grid keys unset the
    /// weight operator uses 99 x 302 and the XC grid its own default;
    /// setting either puts both on the given grid, the unset one at 99 / 302.
    pub grid_radial: Option<usize>,
    /// Lebedev points per radial shell; see `grid_radial`.
    pub grid_angular: Option<usize>,
}

/// One `[[cdft.constraint]]`: a fragment and the population it is held to.
#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct CdftConstraintCfg {
    /// 0-based atom indices of the fragment: non-empty, no repeats.
    pub atoms: Vec<usize>,
    /// Target fragment population in electrons: `N_alpha + N_beta` for
    /// `kind = "charge"` (2.0 on a He atom is neutral He), `N_alpha - N_beta`
    /// for `kind = "spin"`.
    pub target: f64,
    /// `"charge"` (default) or `"spin"`, case-insensitive.
    pub kind: Option<String>,
}

/// The sections a constrained run reads.
const READ_SECTIONS: &[&str] = &[
    "molecule",
    "basis",
    "method",
    "scf",
    "dft",
    "cdft",
    "external_potential",
    "memory",
    "parallel",
    "output",
];

/// The `[scf]` keys a constrained run reads: the inner SCF's convergence and
/// J/K settings that `run_cdft` also takes, plus the process-wide integral
/// settings.
const READ_SCF_KEYS: &[&str] = &[
    "max_iter",
    "energy_conv",
    "density_conv",
    "diis_size",
    "integral_thresh",
    "k_builder",
    "df_j_aux",
    "df_k_aux",
    "ri_j_precision",
    "ri_k_precision",
    "level_shift",
    "mom_after_iter",
    "guess",
    "guess_basis",
    "guess_projection",
    "verbose",
    "eri_precision",
    "jk_storage",
];

fn parse_kind(kind: Option<&str>) -> Result<SpinChannel, String> {
    match kind.map(str::to_ascii_lowercase).as_deref() {
        None | Some("charge") => Ok(SpinChannel::Total),
        Some("spin") => Ok(SpinChannel::SpinDiff),
        Some(other) => Err(format!(
            "[[cdft.constraint]] kind = \"{other}\": valid options are \"charge\" \
             (N_alpha + N_beta on the fragment) and \"spin\" (N_alpha - N_beta)"
        )),
    }
}

fn kind_name(spin: SpinChannel) -> &'static str {
    match spin {
        SpinChannel::Total => "charge",
        SpinChannel::SpinDiff => "spin",
    }
}

/// The value and kind/task rules of `[cdft]` (a pure function of the config;
/// the checks that need the molecule are [`constraints_for`]'s).
pub fn validate(cfg: &Config) -> Result<(), String> {
    let Some(c) = cfg.cdft.as_ref() else {
        return Ok(());
    };
    let kind = cfg.method.kind.as_str();
    if kind != "uhf" {
        return Err(format!(
            "[cdft] is read by method.kind = \"uhf\" only (UHF, or UKS with [dft] functional); \
             kind = \"{kind}\" would not run the constrained solve. A constrained state is \
             solved with unrestricted orbitals even for a singlet."
        ));
    }
    if cfg.method.task != "energy" {
        return Err(format!(
            "[cdft] supports method.task = \"energy\" only (got \"{}\"): there is no \
             constrained gradient, so a geometry or frequency run would follow the \
             unconstrained surface",
            cfg.method.task
        ));
    }
    if cfg.scf.k_builder_is_cosx() {
        return Err("[cdft] does not support [scf] k_builder = \"cosx\"".to_string());
    }
    if cfg.qmmm.is_some() || cfg.cosmo.is_some() || cfg.pcm.is_some() || cfg.cell.is_some() {
        return Err(
            "[cdft] does not combine with [qmmm], [cosmo], [pcm] or [cell]: the constrained \
             solve has no solvation or embedding terms (its Python form takes point charges \
             and a field, which are [external_potential] here)"
                .to_string(),
        );
    }
    if cfg.dft.sets_grid_size()
        || cfg.dft.grid_prune.is_some()
        || cfg.dft.grid_weight_cut.is_some()
        || cfg.dft.dispersion.is_some()
        || cfg.dft.lambda.is_some()
        || cfg.dft.omega.is_some()
    {
        return Err(
            "[cdft] reads [dft] functional only. The grid of the weight operator and of the \
             functional is [cdft] grid_radial / grid_angular, and there is no pruning, \
             weight cut or dispersion on this path"
                .to_string(),
        );
    }
    if c.constraint.is_empty() {
        return Err(
            "[cdft] needs at least one [[cdft.constraint]] with atoms and target".to_string(),
        );
    }
    let mut seen: Vec<(Vec<usize>, SpinChannel)> = Vec::new();
    for (i, k) in c.constraint.iter().enumerate() {
        let spin = parse_kind(k.kind.as_deref())?;
        if k.atoms.is_empty() {
            return Err(format!(
                "[[cdft.constraint]] {i}: atoms must be a non-empty list of atom indices"
            ));
        }
        let mut sorted = k.atoms.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != k.atoms.len() {
            return Err(format!(
                "[[cdft.constraint]] {i}: duplicate atom index in {:?}; a repeated atom would \
                 be weighted twice in the fragment population",
                k.atoms
            ));
        }
        if !k.target.is_finite() {
            return Err(format!(
                "[[cdft.constraint]] {i}: target must be finite, got {}",
                k.target
            ));
        }
        if seen.iter().any(|(a, s)| *a == sorted && *s == spin) {
            return Err(format!(
                "[[cdft.constraint]] {i} repeats the fragment {:?} with kind \"{}\"; two \
                 constraints on the same population make the lambda Jacobian singular",
                k.atoms,
                kind_name(spin)
            ));
        }
        seen.push((sorted, spin));
    }
    if let Some(t) = c.lambda_tol {
        if !(t.is_finite() && t > 0.0) {
            return Err(format!("[cdft] lambda_tol must be finite and > 0, got {t}"));
        }
    }
    if c.max_outer == Some(0) {
        return Err("[cdft] max_outer must be >= 1".to_string());
    }
    cdft_grid(c)?;
    Ok(())
}

/// Every section and `[scf]`/`[dft]` key outside what a constrained run reads
/// is an error (`text` is the raw TOML).
pub fn check_sections(cfg: &Config, text: &str) -> Result<(), String> {
    if cfg.cdft.is_none() {
        return Ok(());
    }
    let raw: toml::Value = toml::from_str(text).map_err(|e| e.to_string())?;
    let written = |name: &str| -> Vec<String> {
        raw.get(name)
            .and_then(|v| v.as_table())
            .map(|t| t.keys().cloned().collect())
            .unwrap_or_default()
    };
    if let Some(top) = raw.as_table() {
        for section in top.keys() {
            if !READ_SECTIONS.contains(&section.as_str()) {
                return Err(format!(
                    "[{section}] is not read by a [cdft] run, which reads [scf] {}, \
                     [dft] functional, [external_potential], [memory], [parallel] and [output]; \
                     remove it",
                    READ_SCF_KEYS.join(" / ")
                ));
            }
        }
    }
    for key in written("scf") {
        if !READ_SCF_KEYS.contains(&key.as_str()) {
            return Err(format!(
                "[scf] {key} is not read by a [cdft] run: the constrained solve is the plain \
                 open-shell SCF inside a lambda loop (no convergence ladder, DF or incremental \
                 pre-stage, SOSCF, TRAH, smearing or symmetry), and reads only [scf] {}",
                READ_SCF_KEYS.join(", ")
            ));
        }
    }
    Ok(())
}

/// `[cdft]` grid keys as the explicit grid of `run_cdft`: `None` when both are
/// unset, else the given size with the unset one at 99 / 302.
fn cdft_grid(c: &CdftCfg) -> Result<Option<ferric_dft::grid::AtomicGridConfig>, String> {
    if c.grid_radial.is_none() && c.grid_angular.is_none() {
        return Ok(None);
    }
    let n_radial = c.grid_radial.unwrap_or(99);
    let n_angular = c.grid_angular.unwrap_or(302);
    if n_radial == 0 {
        return Err("[cdft] grid_radial must be > 0".to_string());
    }
    if !ferric_dft::prune::SUPPORTED_LEBEDEV_ORDERS.contains(&n_angular) {
        return Err(format!(
            "[cdft] grid_angular = {n_angular} is not a supported Lebedev order (supported: {:?})",
            ferric_dft::prune::SUPPORTED_LEBEDEV_ORDERS
        ));
    }
    Ok(Some(ferric_dft::grid::AtomicGridConfig {
        n_radial,
        n_angular,
        ..Default::default()
    }))
}

/// The library constraints of `c` for `mol` (after `apply_ecp`, so `nelec` is
/// the valence electron count). The molecule-dependent checks of Python
/// `run_cdft`: atom indices in range, a charge target in `[0, N_elec]` and a
/// spin target no larger than `N_elec` in magnitude.
fn constraints_for(c: &CdftCfg, mol: &Molecule) -> Result<Vec<Constraint>, String> {
    let natoms = mol.atoms.len();
    let nelec = mol.nelec();
    if nelec <= 0 {
        return Err(format!(
            "[cdft] the molecule has {nelec} electrons; nothing to constrain"
        ));
    }
    let nelec_f = nelec as f64;
    let mut out = Vec::with_capacity(c.constraint.len());
    for (i, k) in c.constraint.iter().enumerate() {
        let spin = parse_kind(k.kind.as_deref())?;
        if let Some(&bad) = k.atoms.iter().find(|&&a| a >= natoms) {
            return Err(format!(
                "[[cdft.constraint]] {i} names atom {bad}, but the molecule has {natoms} atoms \
                 (indices are 0-based)"
            ));
        }
        match spin {
            SpinChannel::Total if k.target < 0.0 || k.target > nelec_f => {
                return Err(format!(
                    "[[cdft.constraint]] {i} (charge) target {} is outside [0, {nelec}]: the \
                     target is a fragment electron POPULATION, not a net charge",
                    k.target
                ));
            }
            SpinChannel::SpinDiff if k.target.abs() > nelec_f => {
                return Err(format!(
                    "[[cdft.constraint]] {i} (spin) target {} exceeds the {nelec} electrons in \
                     the molecule in magnitude",
                    k.target
                ));
            }
            _ => {}
        }
        out.push(Constraint {
            fragment: k.atoms.clone(),
            spin,
            target: k.target,
        });
    }
    Ok(out)
}

/// Run the constrained solve of `cfg.cdft` on the already-built molecule and
/// reference config (`rhf_config` is the run's `RhfConfig`, whose `xc` is the
/// `[dft] functional`), print the result and write the run log's `result` and
/// `run_end` records.
pub fn run(
    cfg: &Config,
    sys: &PreparedSystem,
    bs: &BasisSet,
    rhf_config: &RhfConfig,
) -> Result<(), CliError> {
    let (ctx, mol, prep, bounds) = (sys.ctx(), sys.mol(), sys.basis(), sys.bounds());
    let Some(c) = cfg.cdft.as_ref() else {
        return Err(CliError::msg(
            "internal: cdft::run called without a [cdft] section",
        ));
    };
    let constraints = constraints_for(c, mol).map_err(CliError::msg)?;
    let defaults = RhfConfig::default();
    let config = RhfConfig {
        constraints,
        cdft_lambda_tol: c.lambda_tol.unwrap_or(defaults.cdft_lambda_tol),
        cdft_max_outer: c.max_outer.unwrap_or(defaults.cdft_max_outer),
        cdft_stability_descent: c
            .stability_descent
            .unwrap_or(defaults.cdft_stability_descent),
        dft_grid: cdft_grid(c).map_err(CliError::msg)?,
        ..rhf_config.clone()
    };
    crate::log_jk_path(&config, true);
    let r = ferric_scf::cdft_driver::solve_cdft_uhf(ctx, mol, prep, bs, bounds, &config)
        .map_err(CliError::msg)?;
    let max_constraint_error = r.max_residual();
    let lambda_tol = config.cdft_lambda_tol;
    let scf = r.scf;
    let converged = scf.converged && max_constraint_error < lambda_tol;
    let reference = ferric_scf::certificate::with_scf_envelope(
        crate::scf_certificate_stamped(mol, &scf, &config, None),
        mol,
        &scf,
        &config,
        bs,
    );
    let cert = ferric_scf::certificate::cdft_certificate(
        &reference,
        &config,
        &ferric_scf::certificate::CdftRun {
            lambdas: &r.lambdas,
            populations: &r.populations,
            outer_iterations: r.outer_iters,
            scf_converged: scf.converged,
            max_constraint_error,
        },
    );

    println!(
        "cDFT-{}/{} on {}",
        crate::open_shell_scf_label("UHF", "UKS", &config),
        bs.name,
        cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!("  mult       = {}", mol.multiplicity);
    for (i, k) in config.constraints.iter().enumerate() {
        println!(
            "  constraint {i}: {} of atoms {:?}: target {:.6}, population {:.10}, lambda {:.10}",
            kind_name(k.spin),
            k.fragment,
            k.target,
            r.populations[i],
            r.lambdas[i]
        );
    }
    println!("  outer iterations = {}", r.outer_iters);
    println!("  iterations = {}", scf.iterations);
    println!("  max constraint error = {max_constraint_error:.3e} (lambda_tol {lambda_tol:e})");
    println!("  converged  = {converged}");
    println!("  energy     = {:.10} Hartree", scf.energy);
    if !converged {
        eprintln!(
            "warning: the constrained solve did not converge (inner SCF converged: {}, max \
             constraint error {max_constraint_error:.3e} against lambda_tol {lambda_tol:e})",
            scf.converged
        );
    }
    let cert_json = crate::print_certificate(&cert);
    if let Some(rl) = ferric_scf::runlog::log() {
        rl.result(
            "cdft",
            scf.energy,
            serde_json::json!({
                "lambdas": r.lambdas,
                "populations": r.populations,
                "targets": config.constraints.iter().map(|k| k.target).collect::<Vec<_>>(),
                "kinds": config.constraints.iter().map(|k| kind_name(k.spin)).collect::<Vec<_>>(),
                "residuals": r.residuals,
                "max_constraint_error": max_constraint_error,
                "lambda_tol": lambda_tol,
                "outer_iterations": r.outer_iters,
                "iterations": scf.iterations,
                "scf_converged": scf.converged,
                "converged": converged,
                "certificate": cert_json,
            }),
        );
        rl.run_end(
            scf.energy,
            converged,
            if converged {
                "converged"
            } else {
                "not_converged"
            },
            serde_json::json!({
                "method": cfg.method.kind,
                "task": cfg.method.task,
                "energy_is": "total",
                "scf_iterations": scf.iterations,
            }),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD: &str =
        "[molecule]\nxyz = \"testdata/molecules/lih.xyz\"\n[basis]\nname = \"sto-3g\"\n";

    fn cfg(method: &str, extra: &str) -> Config {
        let src = format!("{HEAD}[method]\n{method}\n{extra}");
        toml::from_str::<Config>(&src).unwrap_or_else(|e| panic!("parse: {e}\n{src}"))
    }

    fn uhf(extra: &str) -> Config {
        cfg("kind = \"uhf\"", extra)
    }

    const ONE: &str = "[cdft]\n[[cdft.constraint]]\natoms = [0]\ntarget = 2.2\n";

    fn refused(c: &Config, needle: &str) {
        let e = validate(c).expect_err("must be refused");
        assert!(e.contains(needle), "message {e:?} lacks {needle:?}");
    }

    #[test]
    fn a_plain_constraint_validates_and_no_section_is_a_no_op() {
        assert!(validate(&uhf(ONE)).is_ok());
        assert!(validate(&uhf("")).is_ok());
    }

    #[test]
    fn the_section_is_read_by_uhf_energy_only() {
        refused(&cfg("kind = \"rhf\"", ONE), "method.kind = \"uhf\" only");
        refused(&cfg("kind = \"ksdft\"", ONE), "method.kind = \"uhf\" only");
        refused(
            &cfg("kind = \"uhf\"\ntask = \"optimize\"", ONE),
            "task = \"energy\" only",
        );
        refused(
            &cfg("kind = \"uhf\"\ntask = \"frequencies\"", ONE),
            "task = \"energy\" only",
        );
    }

    #[test]
    fn constraint_values_are_checked_strictly() {
        refused(&uhf("[cdft]\n"), "at least one [[cdft.constraint]]");
        let one = |body: &str| uhf(&format!("[cdft]\n[[cdft.constraint]]\n{body}\n"));
        refused(&one("atoms = []\ntarget = 1.0"), "non-empty");
        refused(&one("atoms = [0, 0]\ntarget = 1.0"), "duplicate atom index");
        refused(
            &one("atoms = [0]\ntarget = 1.0\nkind = \"Chrage\""),
            "valid options",
        );
        refused(&one("atoms = [0]\ntarget = nan"), "finite");
        // Case-insensitive spelling is accepted, as in Python.
        assert!(validate(&one("atoms = [0]\ntarget = 0.3\nkind = \"SPIN\"")).is_ok());
        let two_same = uhf(
            "[cdft]\n[[cdft.constraint]]\natoms = [1, 0]\ntarget = 1.0\n\
             [[cdft.constraint]]\natoms = [0, 1]\ntarget = 2.0\n",
        );
        refused(&two_same, "repeats the fragment");
        // The same atoms in the other channel is a different constraint.
        let two_kinds = uhf("[cdft]\n[[cdft.constraint]]\natoms = [0]\ntarget = 1.0\n\
             [[cdft.constraint]]\natoms = [0]\ntarget = 0.0\nkind = \"spin\"\n");
        assert!(validate(&two_kinds).is_ok());
    }

    #[test]
    fn loop_and_grid_knobs_are_checked_strictly() {
        let with = |k: &str| {
            uhf(&format!(
                "[cdft]\n{k}\n[[cdft.constraint]]\natoms = [0]\ntarget = 2.2\n"
            ))
        };
        refused(&with("lambda_tol = 0.0"), "lambda_tol");
        refused(&with("lambda_tol = -1e-5"), "lambda_tol");
        refused(&with("max_outer = 0"), "max_outer");
        refused(&with("grid_angular = 111"), "Lebedev");
        refused(&with("grid_radial = 0"), "grid_radial");
        assert!(with("grid_angular = 302").cdft.is_some());
        assert!(validate(&with("lambda_tol = 1e-8\nmax_outer = 5\ngrid_radial = 75")).is_ok());
    }

    #[test]
    fn grid_keys_default_to_99_by_302_for_the_unset_one() {
        let g = |extra: &str| {
            let c = uhf(&format!(
                "[cdft]\n{extra}\n[[cdft.constraint]]\natoms = [0]\ntarget = 2.2\n"
            ));
            cdft_grid(c.cdft.as_ref().unwrap()).unwrap()
        };
        assert!(g("").is_none());
        let a = g("grid_radial = 75").unwrap();
        assert_eq!((a.n_radial, a.n_angular), (75, 302));
        let b = g("grid_angular = 110").unwrap();
        assert_eq!((b.n_radial, b.n_angular), (99, 110));
    }

    #[test]
    fn keys_the_constrained_solve_does_not_read_are_refused() {
        refused(
            &uhf(&format!("[dft]\ngrid_radial = 75\n{ONE}")),
            "grid_radial / grid_angular",
        );
        refused(
            &uhf(&format!(
                "[dft]\nfunctional = \"PBE\"\ndispersion = \"d3bj\"\n{ONE}"
            )),
            "functional only",
        );
        refused(&uhf(&format!("[scf]\nk_builder = \"cosx\"\n{ONE}")), "cosx");
        refused(
            &uhf(&format!("[cosmo]\nepsilon = 78.4\n{ONE}")),
            "does not combine",
        );
        // [dft] functional is the UKS switch and is allowed.
        assert!(validate(&uhf(&format!("[dft]\nfunctional = \"PBE\"\n{ONE}"))).is_ok());
    }

    fn sections(extra: &str) -> Result<(), String> {
        let text = format!("{HEAD}[method]\nkind = \"uhf\"\n{extra}{ONE}");
        let c: Config = toml::from_str(&text).unwrap();
        check_sections(&c, &text)
    }

    #[test]
    fn unread_sections_and_scf_keys_are_refused_at_load() {
        assert!(sections("").is_ok());
        assert!(sections("[scf]\nmax_iter = 99\nlevel_shift = 0.2\nguess = \"hcore\"\n").is_ok());
        let e = sections("[gpu]\nmode = \"off\"\n").unwrap_err();
        assert!(e.contains("[gpu] is not read"), "{e}");
        for key in [
            "soscf = true",
            "stability_descent = true",
            "aurora = true",
            "symmetry = true",
        ] {
            let e = sections(&format!("[scf]\n{key}\n")).unwrap_err();
            assert!(e.contains("is not read by a [cdft] run"), "{key}: {e}");
        }
    }

    /// The shipped inputs pass the full load-time validation, not just the
    /// parser (`all_shipped_examples_parse` stops at serde).
    #[test]
    fn shipped_cdft_and_thole_examples_load() {
        let root = std::env::current_dir()
            .ok()
            .and_then(|cwd| {
                cwd.ancestors()
                    .find(|p| p.join("examples").is_dir() && p.join("testdata").is_dir())
                    .map(std::path::Path::to_path_buf)
            })
            .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."));
        for name in [
            "lih-cdft-charge.toml",
            "lih-plus-cdft-spin.toml",
            "henep-cdft.toml",
            "water-qmmm-thole.toml",
        ] {
            let path = root.join("examples").join(name);
            let c = crate::config::load_config(path.to_str().unwrap())
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(c.cdft.is_some(), name.contains("cdft"), "{name}");
        }
    }

    #[test]
    fn molecule_dependent_checks_mirror_python_run_cdft() {
        let mol = Molecule::parse_xyz("2\nLiH\nLi 0 0 0\nH 0 0 1.6\n", 0, 1).unwrap();
        let c = |body: &str| uhf(&format!("[cdft]\n[[cdft.constraint]]\n{body}\n"));
        let run = |body: &str| constraints_for(c(body).cdft.as_ref().unwrap(), &mol);
        assert!(run("atoms = [1]\ntarget = 1.0").is_ok());
        assert!(run("atoms = [2]\ntarget = 1.0")
            .unwrap_err()
            .contains("names atom 2"));
        // LiH has 4 electrons: a charge target is a population in [0, 4].
        assert!(run("atoms = [0]\ntarget = 4.5")
            .unwrap_err()
            .contains("outside [0, 4]"));
        assert!(run("atoms = [0]\ntarget = -0.1")
            .unwrap_err()
            .contains("outside [0, 4]"));
        assert!(run("atoms = [0]\ntarget = 4.5\nkind = \"spin\"")
            .unwrap_err()
            .contains("exceeds the 4 electrons"));
        let ok = run("atoms = [0]\ntarget = -0.5\nkind = \"spin\"").unwrap();
        assert_eq!(ok[0].spin, SpinChannel::SpinDiff);
        assert_eq!(ok[0].fragment, vec![0]);
        // The target reaches the solver with its sign.
        assert_eq!(ok[0].target, -0.5);
    }
}
