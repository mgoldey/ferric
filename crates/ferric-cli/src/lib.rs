mod config;

use config::{load_config, Config};
/// The `[local]` model types, re-exported for the Python bindings, which
/// apply the SAME rules to their `local=`/`eps=` kwargs.
pub use config::{LocalCfg, LocalDirectKnobs, LocalModel, LocalScheme};
use ferric_cc::ccd::ccd;
use ferric_cc::ccsd::ccsd;
use ferric_cc::ccsd_closed_shell::ccsd_closed_shell;
use ferric_cc::ccsd_t_closed_shell::ccsd_t_closed_shell;
use ferric_cc::double_hybrid::{run_wb97x_l_v, DoubleHybridConfig};
use ferric_cc::linlccd::linlccd;
use ferric_cc::CcConfig;
use ferric_core::basis;
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::att_vv10::{
    att_mp2_vv10, u_att_mp2_vv10, AttVv10Attenuator, AttVv10SpinComponents,
};
use ferric_mp2::attenuated::{attenuated_ri_mp2, AttenuatedMp2Config};
use ferric_mp2::double_hybrid::{mp2_double_hybrid, DoubleHybridKind};
use ferric_mp2::laplace::{laplace_ri_mp2, laplace_sos_mp2, SosFormulation, SosMp2Config};
use ferric_mp2::mp3::mp3_energy;
use ferric_mp2::oo_rimp2::{oo_ri_mp2, OoRiMp2Config};
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_mp2::scs::{scs_mp2, scs_mp2_2terfc, ScsMp2Config, ScsMp2TerfcConfig};
use ferric_rpa::config::{QuadratureConfig, SternheimerConfig};
use ferric_rpa::{run_pdep_rpa, PdepRpaConfig};
use ferric_scf::optimize::{
    optimize_geometry_rohf_with_scf_correction, optimize_geometry_uhf_with_scf_correction,
    optimize_geometry_with_scf_correction, OptimizeConfig,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

fn print_usage() {
    eprintln!("usage: ferric [--verbose|-v] [--json <path>|--no-json] <input.toml>");
    eprintln!();
    eprintln!("Run a ferric quantum-chemistry calculation from a TOML input file.");
    eprintln!(
        "See examples/*.toml for sample inputs and site/src/using/quickstart.md for a walkthrough."
    );
    eprintln!();
    eprintln!("  --verbose, -v   Print one line per SCF iteration to stdout (energy, dE,");
    eprintln!("                  density/DIIS error) as the job runs. Same effect as setting");
    eprintln!("                  `verbose = true` in the [scf] TOML section.");
    eprintln!("  --json <path>   Write the machine-readable JSON Lines run log here.");
    eprintln!("                  Overrides `[output] json`. A run log is written BY");
    eprintln!("                  DEFAULT to <input-stem>.ferric.jsonl beside the input.");
    eprintln!("  --no-json       Do not write a run log. Same as `[output] json = false`.");
}

/// The `method.kind`s graded Proven or Proven (narrow). They never appear in
/// [`EPISTEMIC_WARNINGS`] and print no warning.
///
/// The published grades are the Grade column of the `method.kind` matrix on
/// `site/src/reference/validation.md`; `tests/grades_match_cli_warnings.rs`
/// checks this list and [`EPISTEMIC_WARNINGS`] against that column. A kind in
/// neither list is "not graded": it also prints no warning.
pub const PROVEN_METHOD_KINDS: &[&str] = &[
    "rhf",
    "uhf",
    "rohf",
    "ksdft",
    "rimp2",
    "mp3",
    "att-rimp2",
    "scs-mp2",
    "scs-mp2-2terfc",
    "laplace-mp2",
    "pdep-rpa",
    "ccsd",
    "ccd",
    "ccsd(t)",
    "linlccd",
    "drpa",
    "tda",
    "tddft",
    "oo-rimp2",
    "wb97x-l-v",
];

/// Epistemic-status warnings for `method.kind` values that are graded Smoke
/// or Spike (i.e. NOT Proven / Proven (narrow)).
///
/// SOURCE OF TRUTH: the Grade column of the `method.kind` matrix on
/// `site/src/reference/validation.md` (the wiki's `VALIDATION.md` holds the
/// full record). This table is a condensed, CLI-facing pointer into it, not a
/// second grading system -- when a method's grade changes, update this table,
/// [`PROVEN_METHOD_KINDS`] and the page together;
/// `tests/grades_match_cli_warnings.rs` fails if they disagree. Proven /
/// Proven (narrow) methods are listed in [`PROVEN_METHOD_KINDS`] and never
/// print a warning.
pub const EPISTEMIC_WARNINGS: &[(&str, &str)] = &[
    (
        "gw",
        "method.kind = \"gw\" is Smoke-grade (see site/src/reference/validation.md): QP energies \
         (G0W0@HF/@PBE, frozen core, ECP, U-G0W0@UHF, COHSEX, evGW0, evGW) are compared with PySCF \
         gw_ac/ugw_ac in crates/ferric-gw/tests/validation_gw.rs only at MATCHED settings \
         ([rpa] n_quad = 100, trunc_thresh = 0); the measured agreement is on that page. The CLI \
         defaults (n_quad = 20, trunc_thresh = 1e-4) are coarser: the 20-point frequency grid \
         alone moves the H2O/cc-pVDZ G0W0@PBE HOMO by 13 meV, and truncation is not validated.",
    ),
    (
        "bse-tda",
        "method.kind = \"bse-tda\" is Smoke-grade (see site/src/reference/validation.md): the lowest \
         five singlets match an independent numpy BSE-TDA to 2e-10 Ha given the same \
         quasiparticle energies (H2O cc-pVDZ/aug-cc-pVDZ, NH3 and CH2O cc-pVDZ), but those come from the internal G0W0 at the [rpa] settings of this run, which \
         match PySCF only at n_quad = 100 and trunc_thresh = 0 (the defaults are coarser).",
    ),
    (
        "tdhf-static-polarizability",
        "method.kind = \"tdhf-static-polarizability\" is Smoke-grade (see site/src/reference/validation.md): \
         static alpha is NOT validated -- the one case checked (water/cc-pVDZ, RPAx@PBE, \
         [gw] scissor = 0.36 Ha) gives 5.20 a.u. vs the DOSD reference 9.64, 46% low (an \
         earlier 'matches DOSD' figure came from scissor = 0.0 and a negative alpha diagonal, \
         and is retracted), and the same dense TDHF/RPAx kernel gives C6 ~63% low regardless \
         of gap. At the default scissor = 0.0 this kernel is prone to a genuine excitonic \
         instability that yields a NEGATIVE alpha diagonal; the run hard-errors instead of \
         returning it, so if the job aborts on an unphysical alpha diagonal, set [gw] scissor \
         to ~0.3-0.4 Ha rather than treating it as a crash.",
    ),
    (
        "rs-mp2-rpa",
        "method.kind = \"rs-mp2-rpa\" has Proven energy LIMITS (omega->0/infinity reduce exactly \
         to MP2/MP2+dRPA) but is only Smoke-grade at production omega (see site/src/reference/validation.md): \
         ACONF ties RI-MP2 at omega<=0.3 1/A, and the aug-cc-pVTZ benchmark criterion was met \
         only marginally on one small subset -- treat mid-range-omega numbers as unproven on \
         new systems.",
    ),
    (
        "mp2-v",
        "method.kind = \"mp2-v\" is Smoke-grade (see site/src/reference/validation.md): the damped VV10 \
         half and the erfc-attenuator control match PySCF/numpy references, but the published \
         terfc-attenuated MP2 half has no independent reference, and there \
         is NO comparison to any published MP2-V number (the paper reports only S66/G2 statistics, \
         never a total energy). The defaults (r0 = 1.00 A, b = 11.0, C = 0.0089, terfc, post-HF) \
         are fitted for aug-cc-pVTZ, no counterpoise, frozen core -- running another basis, or \
         with [mp2] frozen_core = 0 (the default here), is unparameterized extrapolation. \
         Open-shell (multiplicity > 1) is DOUBLY unvalidated: S66 is entirely closed-shell, so no \
         open-shell parameterization exists at all.",
    ),
    (
        "b2plyp",
        "method.kind = \"b2plyp\" is Spike-grade: the KS reference uses weighted B88+LYP via \
         per-component XC weights (new infrastructure), and the MP2 correlation reuses the proven \
         RI-MP2 spin-component path, but NO comparison to a reference code exists yet.",
    ),
    (
        "dsd-pbep86",
        "method.kind = \"dsd-pbep86\" is Spike-grade: the KS reference uses weighted PBE+P86 via \
         per-component XC weights (new infrastructure), and the SCS-MP2 correlation reuses the \
         proven RI-MP2 spin-component path, but NO comparison to a reference code exists yet.",
    ),
];

/// Every `method.kind` the CLI dispatches, in the order the unknown-kind
/// error lists them.
///
/// SINGLE SOURCE for both the accept check in [`run`] and the error message
/// ([`unsupported_method_message`]). The two used to be a `matches!` and a
/// hand-written string, and the string drifted: it omitted two kinds the
/// `matches!` accepted. `tests/method_kinds_are_listed.rs`
/// checks this list against the dispatch arms in `run`, so a kind added to
/// one and not the other fails a test.
pub const SUPPORTED_METHOD_KINDS: &[&str] = &[
    "rhf",
    "uhf",
    "rohf",
    "ksdft",
    "rimp2",
    "mp3",
    "oo-rimp2",
    "att-rimp2",
    "mp2-v",
    "scs-mp2",
    "scs-mp2-2terfc",
    "laplace-mp2",
    "laplace-sos-mp2",
    "pdep-rpa",
    "rs-mp2-rpa",
    "gw",
    "bse-tda",
    "tdhf-static-polarizability",
    "ccsd",
    "ccd",
    "ccsd(t)",
    "linlccd",
    "drpa",
    "wb97x-l-v",
    "b2plyp",
    "dsd-pbep86",
    "tda",
    "tddft",
];

/// The error text for an unrecognised `method.kind`, listing every entry of
/// [`SUPPORTED_METHOD_KINDS`] quoted, so it cannot fall out of step with what
/// is accepted.
pub fn unsupported_method_message(method: &str) -> String {
    let listed: Vec<String> = SUPPORTED_METHOD_KINDS
        .iter()
        .map(|k| format!("\"{k}\""))
        .collect();
    format!(
        "unsupported method.kind = \"{method}\"; expected one of {}",
        listed.join(", ")
    )
}

/// Print a one-line epistemic-status warning to stderr if `method` is a
/// Smoke/Stub-grade `method.kind` per `docs/VALIDATION.md`. No-op (and no
/// output) for Proven / Proven (narrow) methods.
fn warn_if_epistemically_unproven(method: &str) {
    if let Some((_, text)) = EPISTEMIC_WARNINGS.iter().find(|(k, _)| *k == method) {
        eprintln!("[warning] {text}");
    }
}

/// Real entry point for the standalone `ferric`/`ferric-cli` binaries. Reads
/// the live process argv, which for a native binary is exactly
/// `[program_name, ...user_args]` -- `run()` below expects that same shape,
/// so this is a one-line adapter, not where the actual logic lives.
pub fn main() {
    run(std::env::args().collect())
}

/// The actual CLI, taking argv explicitly instead of reading
/// `std::env::args()` itself. Split out so a caller whose real OS-process
/// argv does NOT match `[program_name, ...user_args]` can reconstruct that
/// shape and pass it in directly, rather than being stuck with whatever
/// `std::env::args()` happens to return in their process.
///
/// This exists because of a real, measured mismatch: `ferric-python`'s
/// `_cli_main` `#[pyfunction]` (the `ferric` console-script entry point pip
/// installs) is called from INSIDE an already-running Python interpreter,
/// and `std::env::args()` there returns THREE elements for a two-argument
/// invocation -- `[python_interpreter_path, script_path, "--help"]`, not
/// `[script_path, "--help"]` -- confirmed by an inline debug print, not
/// assumed: a real `strace -f -e trace=execve` on the shebang-launched
/// wrapper script showed a completely ORDINARY two-element argv at the
/// process's initial exec, so the extra element is something Python's own
/// runtime does to the argv `std::env::args()`/`/proc/self/cmdline` report
/// AFTER that point, not a shebang or wrapper-script artifact. The old
/// `args[1] == "--help"` check therefore silently checked the WRONG
/// element under the console script (the script's own path, never "--help"
/// or a real TOML path), while working correctly for the native binaries
/// this crate also builds -- caught because `ferric --help`'s exit code
/// (2, not the documented 0) and truncated one-line output differed from
/// the native binary's, not from a logic read of the diff.
///
/// `_cli_main` reconstructs argv from Python's own `sys.argv` (which the
/// pip-generated wrapper script sets correctly -- confirmed against
/// maturin's own docs) instead of trying to normalize `std::env::args()`'s
/// extra element away, since `sys.argv` is the shape Python itself commits
/// to being stable, not an implementation detail of how CPython's runtime
/// happens to report process argv on this platform/version.
pub fn run(args: Vec<String>) {
    // Safe-by-default threading: pin OpenBLAS to 1 thread (rayon owns ferric's
    // parallelism) unless the user explicitly set OPENBLAS_NUM_THREADS. Without
    // this, running the release binary directly oversubscribes rayon × BLAS.
    ferric_integrals::blas_threads::init_threading();
    let ctx = ParallelContext::new();
    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        print_usage();
        std::process::exit(if args.len() < 2 { 2 } else { 0 });
    }
    // Accept the positional TOML path plus an optional `--verbose`/`-v` flag,
    // in either order (`ferric -v input.toml` or `ferric input.toml -v`).
    // `-v`/`--verbose` sets RhfConfig.verbose (live per-iteration SCF
    // progress on stdout) in addition to (not instead of) `[scf] verbose`
    // in the TOML — either one turns it on.
    let mut toml_path: Option<&str> = None;
    let mut cli_verbose = false;
    // JSON run-log overrides from the command line. `None` = defer to
    // `[output] json` in the TOML (which itself defaults to ON).
    let mut cli_json: Option<Option<String>> = None;
    let mut expect_json_path = false;
    for arg in &args[1..] {
        if expect_json_path {
            expect_json_path = false;
            cli_json = Some(Some(arg.clone()));
            continue;
        }
        match arg.as_str() {
            "--verbose" | "-v" => cli_verbose = true,
            "--json" => expect_json_path = true,
            "--no-json" => cli_json = Some(None),
            other if toml_path.is_none() => toml_path = Some(other),
            _ => {
                print_usage();
                std::process::exit(2);
            }
        }
    }
    if expect_json_path {
        eprintln!("error: --json requires a path (use --no-json to disable the run log)");
        std::process::exit(2);
    }
    let Some(toml_path) = toml_path else {
        eprintln!("usage: ferric [--verbose|-v] <input.toml>");
        std::process::exit(2);
    };
    let mut cfg = match load_config(toml_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    cfg.scf.verbose = cfg.scf.verbose || cli_verbose;
    // libint primitive-screening precision for every SCF J/K engine. Process-wide
    // (one CLI run is one job); validated here so a bad value fails before any work.
    if let Err(e) = ferric_integrals::engine_pool::set_eri_precision(cfg.scf.eri_precision) {
        eprintln!("error: [scf] {e}");
        std::process::exit(1);
    }
    eprintln!(
        "ERI precision: {:e}",
        ferric_integrals::engine_pool::eri_precision()
    );

    // Machine-readable JSON run log. ON BY DEFAULT (see `config::OutputCfg`):
    // a result whose run left no artifact cannot be checked afterwards, and
    // this repo has already lost a load-bearing SCF measurement exactly that
    // way. `--json`/`--no-json` override `[output] json`; a path that cannot
    // be opened warns and the run continues without a log, never failing the
    // calculation.
    //
    // Installed HERE -- after the config parses, before anything expensive --
    // so the `run_start` record can carry the resolved config and so every
    // downstream SCF iteration is covered.
    let json_path = match &cli_json {
        Some(explicit) => explicit.as_ref().map(std::path::PathBuf::from),
        None => cfg
            .output
            .resolve_json_path(std::path::Path::new(toml_path)),
    };
    match json_path {
        Some(p) => {
            if ferric_scf::runlog::init(&p) {
                eprintln!("[ferric] JSON run log: {}", p.display());
            }
        }
        // Explicitly poison the sink so a later library call cannot install
        // one the user asked not to have.
        None => ferric_scf::runlog::disable(),
    }

    let method = cfg.method.kind.as_str();
    let task = cfg.method.task.as_str();
    if !SUPPORTED_METHOD_KINDS.contains(&method) {
        eprintln!("error: {}", unsupported_method_message(method));
        std::process::exit(1);
    }
    warn_if_epistemically_unproven(method);
    if !matches!(task, "energy" | "optimize" | "frequencies") {
        eprintln!("error: unsupported method.task = \"{task}\"; expected energy, optimize, or frequencies");
        std::process::exit(1);
    }
    // [dft] dispersion on `optimize` / `frequencies`: the correction is
    // applied inside `run_ksdft`, which only the "energy" task reaches, so each
    // of those tasks applies it itself. Both thread the analytic D3(BJ) or
    // MBD@rsSCS gradient through a closure that sees the converged SCF at every
    // geometry (`optimize_geometry_with_scf_correction`,
    // `harmonic_frequencies_with_scf_correction`), so the energy, the gradient
    // and the finite-difference Hessian all describe one surface. UKS
    // `optimize` takes the same correction through
    // `optimize_geometry_uhf_with_scf_correction` (MBD@rsSCS with the
    // unrestricted Z-vector) and ROKS `optimize` through
    // `optimize_geometry_rohf_with_scf_correction` (the ROKS Z-vector);
    // open-shell `frequencies` are refused
    // (`refuse_open_shell_dispersion_gradient`).
    // ...and the same for the METHOD, which the task guard above does not
    // cover. The correction is evaluated only where a Kohn-Sham SCF result is
    // printed (`print_scf_energy`), so a plain `rhf` energy run passes the
    // task check, dispatches to `run_rhf`, and never sees the dispersion key
    // at all -- reporting a plain HF energy from a config that asks for a
    // corrected one. There is no correct answer to substitute either: D3(BJ)'s
    // damping parameters and MBD@rsSCS's beta are fitted PER FUNCTIONAL, so
    // there is no such thing as "D3(BJ) for Hartree-Fock" without naming a
    // fit. Any KS SCF qualifies: `ksdft`, or `rhf`/`uhf`/`rohf` with `[dft] functional`.
    if cfg.dft.dispersion.is_some() && cfg.ks_functional().is_none() {
        eprintln!(
            "error: [dft] dispersion is only supported on a Kohn-Sham SCF (method.kind = \
             \"ksdft\", or rhf/uhf/rohf with [dft] functional); got kind = \"{method}\" \
             without one. Dispersion is evaluated on the KS-DFT path only, so this run would \
             silently report an UNCORRECTED energy. Its parameters (D3(BJ) damping, MBD@rsSCS \
             beta) are fitted per functional, so there is no default fit to apply here -- \
             remove the dispersion key, or use kind = \"ksdft\"."
        );
        std::process::exit(1);
    }
    // [dft] grid_prune — main-DFT-grid angular pruning. Strict parse (unknown
    // values are a hard error, never a silent default), then a task guard:
    // the XC gradient's grid-response term is only built for the unpruned
    // grid, so optimize/frequencies must refuse a pruned grid up front rather
    // than returning a gradient that is not the gradient of the energy the
    // SCF converged.
    let grid_prune: Option<ferric_dft::prune::PruneScheme> = match cfg.dft.grid_prune.as_deref() {
        None => None,
        Some(s) => match ferric_dft::prune::PruneScheme::parse_config_str(s) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("error: [dft] grid_prune: {e}");
                std::process::exit(1);
            }
        },
    };
    if grid_prune == Some(ferric_dft::prune::PruneScheme::Sgx) {
        eprintln!(
            "error: [dft] grid_prune = \"sgx\" is the COSX exchange grid's scheme \
             ([scf] cosx_grid); the XC grid is validated with \"nwchem\" or \"none\" only."
        );
        std::process::exit(1);
    }
    if grid_prune.is_some() && task != "energy" {
        eprintln!(
            "error: [dft] grid_prune is supported for method.task = \"energy\" only \
             (got \"{task}\"). The XC gradient's grid-response term is built on the \
             unpruned grid, so a pruned energy and its gradient would be inconsistent."
        );
        std::process::exit(1);
    }
    // Any other `[dft]` key the selected kind never reads (a functional on
    // uhf/rhf/rimp2..., lambda/omega off wb97x-l-v, a grid_prune with no KS
    // grid) is refused rather than silently dropped. See
    // `Config::validate_dft_section`.
    if let Err(e) = cfg.validate_dft_section() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
    // ...and the same for keys a TASK path never reads (see
    // `Config::validate_task_compat`).
    if let Err(e) = cfg.validate_task_compat() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
    // QM/MM: the QM region becomes the molecule that is solved, and the MM
    // region becomes the external potential it is solved in. Built BEFORE the
    // molecule so the two cannot disagree about which atoms are quantum --
    // when `[qmmm]` is present, `[molecule].xyz` is not read at all.
    let (qmmm_system, mut mol) = resolve_geometry(&cfg);
    // Every `on {}` header and the run log's molecule `path` read
    // `cfg.molecule.xyz`. With `[qmmm]` that file was NOT read -- the geometry
    // came from the PQR -- so they were naming a file that did not produce the
    // atoms being solved. Point the field at the real source rather than
    // threading a second one through ~20 print sites: with `[qmmm]` present
    // the PQR IS `molecule.xyz`'s job.
    //
    // This is not cosmetic. The stale header is what makes the vacuum
    // reference easy to get wrong: the run says `on water.xyz` while solving
    // the PQR geometry, so re-running without `[qmmm]` looks like the same
    // molecule and is not (see examples/water-qmmm.toml).
    if let Some(q) = cfg.qmmm.as_ref() {
        cfg.molecule.xyz = q.pqr.clone();
    }
    // Closed-shell-only kinds refuse an open-shell molecule HERE, before any
    // integral: otherwise an odd electron count dies inside the SCF as
    // "ScfConvergence { iterations: 0 }" and an even one (triplet water)
    // used to come back as the singlet. See `Config::validate_multiplicity`.
    if let Err(e) = cfg.validate_multiplicity(mol.multiplicity) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
    let bs = if let Some(name) = &cfg.basis.name {
        basis::bundled(name)
    } else if let Some(path) = &cfg.basis.path {
        basis::load_g94(path)
    } else {
        Err(ferric_core::FerricError::Basis("no basis specified".into()))
    }
    .unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });

    // Populate per-atom n_core_ecp when the basis carries ECPs (no-op otherwise),
    // so nelec()/nuclear_repulsion() use the effective valence electron count and
    // charge. PreparedBasis::new derives the effective nuclear charge directly
    // from bs.ecps, so this must happen before any nelec()-derived occupation.
    mol.apply_ecp(&bs);

    // `frozen_core = "auto"` becomes a number HERE and nowhere earlier: the
    // count depends on the molecule AND on the basis, because an ECP has
    // already removed some core orbitals from the MO space (apply_ecp, just
    // above, is what puts those counts on the atoms). Validate every
    // correlation section's key up front so an over-large frozen core is an
    // error before the SCF runs, not a bare message from inside the
    // correlation kernel 40 seconds later, and print what "auto" resolved to
    // so the run's correlation space is auditable from its log alone.
    if let Err(e) = cfg.validate_frozen_core(&mol) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
    for line in cfg.frozen_core_audit_lines(&mol) {
        eprintln!("[ferric] {line}");
    }

    let prep = PreparedBasis::new(&mol, &bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let op = Operator::coulomb();
    // Resolved ONCE, here, so the SAME kind governs BOTH mechanisms:
    //   * this `bounds` value, whose `csb_m` table (attached by
    //     `compute_for_screening`) is what carries CSB into the default
    //     `DirectJ`/`DirectK`/`DirectJK`/`build_jk` path for RHF, UHF and
    //     ROHF — none of whose signatures change;
    //   * `RhfConfig::screening` below, which governs the separate LinK path.
    // Parsing it twice would risk the two silently disagreeing after a future
    // edit touched only one site.
    let screening_kind = cfg
        .scf
        .screening
        .as_deref()
        .map_or(Ok(ferric_scf::screening::ScreeningKind::default()), |s| {
            ferric_scf::screening::ScreeningKind::parse_config_str(s)
        })
        .unwrap_or_else(|e| {
            eprintln!("error: [scf] screening: {e}");
            std::process::exit(1);
        });
    let bounds =
        SchwarzBounds::compute_for_screening(op, &prep, screening_kind).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });
    // The SCF functional and the RI-J/RI-K aux defaults for this kind (any
    // Kohn-Sham SCF, including rhf/uhf/rohf promoted by `[dft] functional`,
    // gets RI-JK via def2-universal-jkfit). See
    // `Config::scf_xc_and_aux_defaults` for the per-kind table and why.
    let (xc, df_j_default, df_k_default) = cfg.scf_xc_and_aux_defaults();
    refuse_tddft_xc_without_kernel(&cfg, method);
    // Unified memory budget from [memory] (bytes), threaded into EVERY method
    // config below. `None` → each method's resolver auto-detects (0.8 × RAM).
    // Log the resolved value + source once, up front, so runs are auditable.
    let budget_bytes: Option<usize> = cfg.memory.budget_bytes();
    {
        let resolution = ferric_core::memory::resolve_budget(budget_bytes);
        // Install ONE process-global pool from the SAME resolution that the
        // audit line reports, so the printed ceiling and the enforced ledger
        // can never disagree.
        //
        // Why a pool and not just this number: before this, every gate
        // compared its own plane against 100% of this figure and they all
        // passed, so the process held the SUM. The measured case was a
        // 27-atom def2-SVP B3LYP RI-JK single point that printed
        // `memory budget: 4.72 GiB` and then died at MAXRSS 6.04 GiB to a
        // global OOM kill without one gate failing: the DF 3-index tensor
        // asked "do I fit in 4.72 GiB?" (yes), the grid AO cache asked the
        // same question against the same full number (yes). A pool is
        // DEBITED, so whichever asks second sees only what the first left.
        //
        // Installed unconditionally (auto-detected budgets included) because
        // the composition defect is not specific to an explicit budget — the
        // OOM above happened on an auto-detected one. Every gate's no-pool
        // behaviour is still the trivial limit for library callers that never
        // install one (pinned by
        // ferric-core/tests/mwe_pool_no_budget_is_a_noop.rs and
        // ferric-scf/tests/mwe_ksdft_pool_is_inert_without_a_pool.rs).
        let pool = ferric_core::memory::pool::MemoryPool::with_capacity_bytes(resolution.bytes);
        ferric_core::memory::pool::install_global(pool);
        eprintln!("[ferric] {}", resolution.audit_line());
    }
    let rhf_config = RhfConfig {
        xc_omega: None,
        max_iter: cfg.scf.max_iter,
        energy_conv: cfg.scf.energy_conv,
        density_conv: cfg.scf.density_conv,
        diis_size: cfg.scf.diis_size,
        diis_flavor: cfg.scf.diis_flavor().unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        }),
        diis_switch_thresh: cfg.scf.diis_switch_thresh.unwrap_or(1e-1),
        smearing_sigma: cfg.scf.smearing_sigma,
        integral_thresh: cfg.scf.integral_thresh,
        k_builder: cfg.scf.k_builder.clone(),
        cosx: cfg.scf.cosx_config().unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        }),
        // Shared spelling parser: "exact"/"none"/"off"/"conventional" mean
        // the same "" (no density fitting) as in the Python bindings.
        df_j_aux: cfg.scf.df_j_aux_resolved().or(df_j_default),
        // RIJCOSX: `k_builder = "cosx"` replaces RI-K, so the RI-K default
        // is not applied under it (an explicit `df_k_aux` next to COSX is
        // refused by the SCF as a conflict).
        df_k_aux: cfg
            .scf
            .df_k_aux_resolved()
            .or(if cfg.scf.k_builder.as_deref() == Some("cosx") {
                None
            } else {
                df_k_default
            }),
        xc,
        // `None` keeps `AtomicGridConfig::default()` (75x110, unpruned) —
        // byte-identical to the historical path. Only a `[dft] grid_prune`
        // setting materialises an explicit config, and it touches the MAIN
        // grid only: `nlc_grid` stays `None` so the VV10/NLC grid keeps its
        // 50x50 unpruned default, where pruning has no valid table.
        // `[dft] grid_radial`/`grid_angular` size the same main grid (unset
        // sizes keep the 75x110 default); all three keys unset is `None`.
        dft_grid: cfg.dft.grid_config(grid_prune).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        }),
        nlc_grid: None,
        level_shift: cfg.scf.level_shift.unwrap_or(0.0),
        newton_trigger: if cfg.scf.soscf { 1e-3 } else { 0.0 },
        ah_trigger: 0.0,
        mom_after_iter: cfg.scf.mom_after_iter,
        // ROHF/ROKS F6 occupation guard (ferric_scf::rohf_occupation): ON, as
        // in the Default impl. Not a TOML key; listed because this literal is
        // exhaustive.
        rohf_occupation_guard: true,
        constraints: Vec::new(),
        cdft_lambda_tol: 1e-5,
        // cDFT is not CLI-wired (constraints above are always empty), so this
        // is inert here; it is listed only because the literal is exhaustive.
        cdft_max_outer: 30,
        cdft_lambda_init: None,
        // SCF accelerators, both opt-in and OFF here. Neither is CLI-wired
        // yet; these are listed only because the literal is exhaustive.
        //
        // AURORA is measured to win on closed-shell RHF (-24.2% J/K builds,
        // growing with size) but is NOT defaulted on: its evidence is
        // closed-shell RHF only, pure functionals are known-weak, open shell
        // is unimplemented, and the paper's D_k^xc term -- the reason pure
        // functionals fail -- is named but never defined.
        //
        // TRAH wins ITERATIONS everywhere measured but loses WALL TIME
        // everywhere (45x DIIS on benzene/cc-pVDZ), because each step pays a
        // Davidson of Fock builds. Its value is convergence RESCUE, not
        // throughput.
        aurora: Default::default(),
        trah: Default::default(),
        trah_trigger: None,
        // Inert on this path (`constraints` is empty, so `solve_cdft_uhf` is
        // never reached), but spelled out rather than left to a `..default()`
        // that this literal does not use — a struct literal that lists every
        // field is how a new knob gets noticed here instead of silently
        // acquiring whatever the Default impl says.
        cdft_stability_descent: true,
        // `[scf] stability_descent` (rhf, uhf and ksdft; other kinds were
        // refused by `Config::validate_cli_wired_keys`). It needs the
        // stability verdict, so it turns `check_stability` on too, exactly as
        // the Python `run_uhf(stability_descent=True)` does.
        scf_stability_descent: cfg.scf.stability_descent,
        fractional_occ: false,
        // 0 = "unset" → the SCF resolver auto-detects (0.8×RAM). An explicit
        // [memory] budget (incl. a deliberate 2 GiB) is passed through and honored.
        three_index_budget_bytes: budget_bytes.unwrap_or(0),
        init_guess_density: None,
        use_sad_guess: cfg.scf.use_density_guess().unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        }),
        stall_window: None,
        divergence_tol: None,
        // A `[qmmm]` MM region and an explicit `[external_potential]` are two
        // sources for the same field. Combining them silently would double a
        // contribution nobody asked for, so the QM/MM one wins and the clash
        // is refused above.
        external_potential: match &qmmm_system {
            Some(sys) => sys.to_external_potential(),
            None => cfg.external_potential.to_external_potential(),
        },
        cosmo: cfg.cosmo.clone(),
        // `[pcm]`: IEF-PCM. Values were validated at load time and the
        // kind/task scope by `Config::validate_cli_wired_keys`.
        pcm: resolve_pcm(&cfg),
        // Polarizable (Thole) embedding has no TOML surface either -- it is
        // reachable from Rust (RhfConfig.polarizable) and Python
        // (QmmmSystem(polarizabilities_angstrom3=) + run_qmmm) only.
        polarizable: None,
        verbose: cfg.scf.verbose,
        check_stability: cfg.scf.runs_stability_check(),
        // Same resolved kind that already selected `bounds`'s CSB table
        // above; see the comment there for why it is parsed once.
        screening: screening_kind,
    };

    // Header record for the JSON run log: everything needed to reproduce this
    // run, written before any expensive work so it survives even a job killed
    // in the first SCF iteration. No-op when no log is installed.
    if let Some(rl) = ferric_scf::runlog::log() {
        rl.run_start(
            serde_json::json!({
                "method": method,
                "task": task,
                "basis": bs.name,
                "functional": cfg.dft.functional,
                "xc": rhf_config.xc,
                "max_iter": cfg.scf.max_iter,
                "energy_conv": cfg.scf.energy_conv,
                "density_conv": cfg.scf.density_conv,
                "df_j_aux": rhf_config.df_j_aux,
                "df_k_aux": rhf_config.df_k_aux,
                "level_shift": rhf_config.level_shift,
                // The RESOLVED budget in bytes, not the raw `[memory]` key:
                // an omitted budget auto-detects, and the number actually used
                // is the one a post-mortem reader needs.
                "memory_budget_bytes": ferric_core::memory::resolve_budget_bytes(budget_bytes),
                "openblas_num_threads": std::env::var("OPENBLAS_NUM_THREADS").ok(),
                "rayon_num_threads": rayon::current_num_threads(),
                "mpi_ranks": ctx.size,
            }),
            serde_json::json!({
                "path": cfg.molecule.xyz,
                "n_atoms": mol.atoms.len(),
                "formula": molecular_formula(&mol),
                "charge": cfg.molecule.charge,
                "multiplicity": cfg.molecule.multiplicity,
                "n_electrons": mol.nelec(),
                "n_basis": prep.nbasis(),
            }),
        );
    }

    // Resolve/validate [scf] df_guess_aux up front (config-honesty: a knob
    // that would silently do nothing under df_guess = false is a hard error)
    // so a typo'd TOML fails fast instead of after an expensive SCF.
    let df_guess_aux = cfg.scf.df_guess_aux_resolved().unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let df_increments_aux = cfg.scf.df_increments_aux_resolved().unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });

    // From here on `method` is the kind to DISPATCH on (the run-log header
    // above recorded the kind as written): `ksdft` on an open-shell molecule
    // runs the UKS route under "uhf", and `rhf` with `[dft] functional` runs
    // RKS under "ksdft". `rhf_config.xc` already carries the functional, which
    // is what makes the uhf/rohf solvers and gradients UKS/ROKS. See
    // `Config::dispatch_kind`.
    let method = cfg.dispatch_kind(mol.multiplicity);

    if task == "optimize" {
        run_optimize(method, &cfg, &ctx, &mol, &bs, op, &rhf_config, budget_bytes);
        return;
    }

    if task == "frequencies" {
        run_frequencies(method, &cfg, &ctx, &mol, &bs, op, &rhf_config);
        return;
    }

    if method == "uhf" {
        run_uhf(&cfg, &ctx, &mol, &bs, op, &prep, &bounds, &rhf_config);
        return;
    }

    if method == "rohf" {
        run_rohf(&cfg, &ctx, &mol, &bs, op, &prep, &bounds, &rhf_config);
        return;
    }

    // Open-shell requests for closed-shell-only kinds (linlccd, the double
    // hybrids, tda/tddft, ...) were refused up front by
    // `Config::validate_multiplicity`, right after the geometry was resolved;
    // the open-shell correlated kinds it admits get a UHF reference from the
    // shared SCF below (`solve_open_shell_reference`).

    // The wB97X-L-V double hybrid converges its OWN Kohn-Sham reference inside
    // `run_wb97x_l_v` (it forces `xc = "wB97X-L-V"` and runs `ksdft_ladder`), so
    // it returns early here rather than falling through to the unconditional
    // `solve_rhf` below. Letting it fall through would run a full plain-HF SCF
    // whose result is then thrown away, and — worse — the reference actually
    // consumed would silently be the wrong one.
    if method == "wb97x-l-v" {
        run_wb97x_l_v_arm(
            &cfg,
            &ctx,
            &mol,
            &bs,
            &prep,
            &bounds,
            &rhf_config,
            budget_bytes,
        );
        return;
    }
    if matches!(method, "b2plyp" | "dsd-pbep86") {
        run_mp2_double_hybrid_arm(
            &cfg,
            &ctx,
            &mol,
            &bs,
            &prep,
            &bounds,
            &rhf_config,
            budget_bytes,
            method,
        );
        return;
    }

    // An exact `drpa` that cannot fit is refused here, before the SCF.
    preflight_exact_drpa(&cfg, &mol, &prep, budget_bytes);

    // RHF and closed-shell KS-DFT both run through `solve_rhf` (KS-DFT is
    // `solve_rhf` with `cfg.xc` set), so both take the level-shift ladder. This
    // gives KS-DFT the same DIIS-oscillation fallback RHF already had: a hybrid
    // like B3LYP on a π-system that limit-cycles at level_shift=0 escalates the
    // virtual-block shift instead of silently running to max_iter. `build_ladder`
    // (config.rs) dispatches on `base.xc` -- KS-DFT runs get `ksdft_ladder`
    // (starts from the caller's own max_iter, carries the DFT grid), plain RHF
    // gets `default_ladder_from` (hard-codes DF-JK, HF-tuned rung budgets). See
    // docs/profiles-2026-07-14.md finding (2) + its 2026-07-19 correction note
    // (the CLI's `ksdft` path previously fell through to the HF-tuned ladder,
    // which starves rung 0 of iterations and walked the whole ladder to
    // MaxIter instead of converging) + the ksdft_ladder tests in
    // ferric-scf/src/ladder.rs.
    let result = if method == "rhf" || method == "ksdft" {
        // `[scf] df_guess` is not (yet) composed with the multi-rung
        // convergence ladder — the ladder's own escalation (level shift /
        // ADIIS / smearing) is a different mechanism aimed at hard-to-
        // converge systems, and interleaving a DF pre-stage into every rung
        // needs its own design (which rung(s) get it, whether DIIS state
        // should reset between the ladder's *own* rungs the same way). Warn
        // rather than silently ignore, per the config-honesty convention.
        // Only an EXPLICIT `df_guess = true`: the knob defaults on, so
        // df_guess_enabled() would warn on every plain rhf/ksdft run.
        if cfg.scf.df_guess == Some(true) {
            eprintln!(
                "warning: [scf] df_guess is not yet composed with the {method} convergence ladder; ignored here (use kind = \"rimp2\" or another non-laddered method to use it)"
            );
        }
        if cfg.scf.df_increments {
            eprintln!(
                "warning: [scf] df_increments is not yet composed with the {method} convergence ladder; ignored here (use kind = \"rimp2\" or another non-laddered method to use it)"
            );
        }
        let ladder = cfg.scf.build_ladder(&rhf_config).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });
        // Report the J/K path actually in use. RI-JK is now opt-in (the ladder
        // no longer substitutes it — see `ladder::default_ladder_from`), but
        // KS-DFT still auto-selects an aux above, so state which one ran rather
        // than leaving a CLI-vs-library energy comparison to guesswork.
        if let Some(rung0) = ladder.first() {
            log_jk_path(&rung0.config, false);
        }
        let lr = ferric_scf::ladder::solve_rhf_ladder(&ctx, &mol, &prep, op, &bounds, &ladder)
            .unwrap_or_else(|e| {
                eprintln!("error: SCF ladder failed: {e:?}");
                std::process::exit(1);
            });
        if !lr.converged {
            eprintln!(
                "warning: SCF did not fully converge (best rung {}, exit {:?})",
                lr.rung_reached,
                lr.rung_outcomes.last().map(|o| o.exit)
            );
        }
        lr.result
    } else if mol.multiplicity > 1 {
        // An open-shell molecule reaching here passed
        // `Config::validate_multiplicity`, so its kind has an open-shell
        // route (rimp2/oo-rimp2/pdep-rpa/gw/mp2-v on task = "energy"); each
        // consumes a UHF reference. `solve_rhf` would refuse the molecule.
        // GW picks its own reference (`[gw] reference = "uhf" | "rohf"`); it
        // is solved once here and `run_gw` reuses it, so a UHF failure can
        // never end a run that asked for ROHF.
        if method == "gw" {
            gw_open_shell_reference(&cfg, &ctx, &mol, &prep, op, &bounds, &rhf_config)
                .map(|(r, _)| r)
                .unwrap_or_else(|e| {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                })
        } else {
            solve_open_shell_reference(method, &cfg, &ctx, &mol, &prep, &bounds, &rhf_config)
        }
    } else if cfg.scf.df_guess_enabled() {
        // Opt-in DF-guess two-stage SCF (see
        // `ferric_scf::ladder::solve_rhf_with_df_guess`). Closed-shell only:
        // open-shell (UHF/ROHF) df_guess is not implemented, and open-shell
        // molecules took the branch above (which warns that the key is
        // ignored there).
        let dfg = ferric_scf::ladder::solve_rhf_with_df_guess(
            &ctx,
            &mol,
            &prep,
            op,
            &bounds,
            &rhf_config,
            df_guess_aux.as_deref(),
        )
        .unwrap_or_else(|e| {
            eprintln!("error: DF-guess SCF failed: {e:?}");
            std::process::exit(1);
        });
        eprintln!(
            "[ferric] SCF: DF-guess pre-stage via {} ({} iters, {}), exact stage from that density",
            df_guess_aux
                .as_deref()
                .unwrap_or(ferric_scf::ladder::DF_GUESS_DEFAULT_AUX),
            dfg.df_iterations,
            if dfg.df_converged {
                "converged"
            } else {
                "did not fully converge"
            },
        );
        let _ = dfg.df_energy; // diagnostic only; the exact-stage result is authoritative
        dfg.result
    } else if cfg.scf.df_increments {
        // Opt-in DF-corrected incremental Fock SCF (see
        // `ferric_scf::df_increments::solve_rhf_with_df_increments`).
        // Closed-shell only, for the same reason `df_guess` is above.
        let dfi = ferric_scf::df_increments::solve_rhf_with_df_increments(
            &ctx,
            &mol,
            &prep,
            op,
            &bounds,
            &rhf_config,
            df_increments_aux.as_deref(),
        )
        .unwrap_or_else(|e| {
            eprintln!("error: DF-increments SCF failed: {e:?}");
            std::process::exit(1);
        });
        eprintln!(
            "[ferric] SCF: DF-increments via {} (DF-guess {} iters {}; DF-corrected inner loop {} iters {}; \
             {} exact cleanup iters; {} total exact Fock builds)",
            df_increments_aux.as_deref().unwrap_or(ferric_scf::ladder::DF_GUESS_DEFAULT_AUX),
            dfi.df_guess_iterations,
            if dfi.df_guess_converged { "converged" } else { "did not fully converge" },
            dfi.inner_iterations,
            if dfi.inner_converged { "converged" } else { "did not fully converge" },
            dfi.exact_cleanup_iterations,
            dfi.exact_builds,
        );
        if !dfi.result.converged {
            eprintln!(
                "warning: DF-increments SCF did not fully converge (exit {:?})",
                dfi.result.exit
            );
        }
        let _ = dfi.df_guess_energy; // diagnostic only; the final exact result is authoritative
        dfi.result
    } else {
        solve_rhf(&ctx, &mol, &prep, op, &bounds, &rhf_config).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        })
    };

    // Same-basis Hirshfeld proatom: neutral free-atom SCF densities in the
    // molecule's OWN basis and SCF settings, built lazily per call. Shared by
    // all Hirshfeld consumers (charges, effective volumes, per-atom
    // polarizability) and by the Python `ferric.hirshfeld_charges` default.
    let proatom = ferric_scf::properties::scf_proatom_provider(&ctx, &bs, op, &rhf_config);

    // Snapshot the scalars the terminal log record needs BEFORE the dispatch:
    // one arm (`run_pdep_rpa_arm`) takes `result` by value. Three `Copy`
    // fields, so this is free -- cloning the whole `ScfResult` (several dense
    // nbasis x nbasis matrices) to satisfy the borrow checker would be a real
    // allocation added by logging, which is exactly what must not happen.
    let scf_energy = result.energy;
    let scf_converged = result.converged;
    let scf_exit = result.exit;
    let scf_iterations = result.iterations;
    let cosx_final = result.cosx_final;
    // `[dft] dispersion` is admitted only on a Kohn-Sham SCF (guarded above),
    // and the closed-shell KS route dispatches as "ksdft". Evaluated ONCE here
    // and shared by the printout and the terminal `run_end` record.
    let dispersion = if method == "ksdft" {
        dispersion_correction(
            &cfg,
            &ctx,
            &mol,
            &bs,
            op,
            &rhf_config,
            result.density_total(),
        )
    } else {
        None
    };

    match method {
        "rhf" => run_rhf(&cfg, &bs, &prep, &result),
        "ksdft" => run_ksdft(&cfg, &bs, &prep, &result, dispersion.as_ref()),
        "rimp2" => run_rimp2(&cfg, &mol, &bs, &prep, op, &result, budget_bytes),
        "mp3" => run_mp3(&cfg, &mol, &bs, &prep, op, &result, budget_bytes),
        "oo-rimp2" => run_oo_rimp2(
            &cfg,
            &mol,
            &bs,
            &prep,
            op,
            &bounds,
            &result,
            budget_bytes,
            rhf_config.external_potential.as_ref(),
        ),
        "att-rimp2" => run_att_rimp2(&cfg, &mol, &bs, &prep, &result, budget_bytes),
        "mp2-v" => run_mp2_v(&cfg, &mol, &bs, &prep, &result, budget_bytes),
        "rs-mp2-rpa" => run_rs_mp2_rpa(&cfg, &mol, &bs, &prep, &result, budget_bytes),
        "scs-mp2" => run_scs_mp2(&cfg, &mol, &bs, &prep, &result, budget_bytes),
        "scs-mp2-2terfc" => run_scs_mp2_2terfc(&cfg, &mol, &bs, &prep, &result, budget_bytes),
        "ccsd" => run_ccsd(&cfg, &mol, &bs, &prep, op, &result, budget_bytes),
        "linlccd" => run_linlccd(&cfg, &mol, &bs, &prep, op, &result, budget_bytes),
        "ccd" => run_ccd(&cfg, &mol, &bs, &prep, op, &result, budget_bytes),
        "ccsd(t)" => run_ccsd_t(&cfg, &mol, &bs, &prep, op, &result, budget_bytes),
        "drpa" => run_drpa(&cfg, &mol, &bs, &prep, op, &result, budget_bytes),
        "laplace-mp2" => run_laplace_mp2(&cfg, &mol, &bs, &prep, op, &result, budget_bytes),
        "laplace-sos-mp2" => run_laplace_sos_mp2(&cfg, &mol, &bs, &prep, op, &result, budget_bytes),
        "pdep-rpa" => run_pdep_rpa_arm(
            &cfg,
            &ctx,
            &mol,
            &bs,
            &prep,
            op,
            &bounds,
            &rhf_config,
            result,
            budget_bytes,
            &proatom,
        ),
        "gw" => run_gw(&cfg, &mol, &bs, &prep, op, &result, budget_bytes),
        "bse-tda" => run_bse_tda(&cfg, &mol, &bs, &prep, op, &result, budget_bytes),
        "tdhf-static-polarizability" => {
            run_tdhf_static_polarizability(&cfg, &mol, &bs, &prep, op, &result, budget_bytes)
        }
        "tda" | "tddft" => run_tddft_arm(
            &cfg,
            &mol,
            &bs,
            &prep,
            &result,
            budget_bytes,
            method,
            &rhf_config,
        ),
        _ => unreachable!(),
    }

    // Terminal record: the SCF result plus wall/cpu time and peak RSS.
    //
    // The energy reported here is the SCF energy, which for a post-SCF method
    // (MP2/RPA/CC/GW) is the REFERENCE, not the method's total -- and the
    // record says so via `energy_is`, rather than labelling a reference energy
    // as the run's answer. The method's OWN total goes in a separate `result`
    // record emitted by that method's branch.
    //
    // Methods not yet wired emit `result_unlogged` instead of nothing, so a
    // consumer can distinguish "this method does not log its result yet" from
    // "this run died before producing one". Keep RESULT_LOGGED in step with
    // the `rl.result(...)` call sites; a name here that has no matching call
    // site would claim coverage that does not exist.
    const RESULT_LOGGED: &[&str] = &[
        "rimp2",
        "oo-rimp2",
        "att-rimp2",
        "laplace-mp2",
        "laplace-sos-mp2",
        "scs-mp2",
        "scs-mp2-2terfc",
        "mp3",
        "ccsd",
        "ccd",
        "ccsd(t)",
        "linlccd",
        "drpa",
        "mp2-v",
        "rs-mp2-rpa",
    ];
    let scf_only = matches!(method, "rhf" | "uhf" | "rohf" | "ksdft");
    if !scf_only && !RESULT_LOGGED.contains(&method) {
        if let Some(rl) = ferric_scf::runlog::log() {
            rl.result_unlogged(method);
        }
    }
    if let Some(rl) = ferric_scf::runlog::log() {
        let mut extra = serde_json::json!({
            // The kind as written, like the `run_start` header (`method`
            // is the dispatch kind by now: `rhf` + functional reads
            // "ksdft" there).
            "method": cfg.method.kind,
            "task": task,
            "scf_iterations": scf_iterations,
            "energy_is": if method == "rhf" || method == "uhf" || method == "rohf" || method == "ksdft" {
                "total"
            } else {
                "scf_reference_only"
            },
        });
        // With a dispersion correction, `energy` is the corrected TOTAL (what
        // `energy_is: "total"` promises) and the SCF energy and the correction
        // are recorded alongside it. Without one, no key is added.
        let energy = match (&dispersion, extra.as_object_mut()) {
            (Some(d), Some(obj)) => {
                obj.insert("scf_energy".to_string(), serde_json::json!(scf_energy));
                obj.insert("dispersion".to_string(), d.to_json());
                scf_energy + d.energy
            }
            _ => scf_energy,
        };
        if let (Some(f), Some(obj)) = (cosx_final, extra.as_object_mut()) {
            obj.insert("cosx_final_pass".to_string(), cosx_final_json(&f));
        }
        rl.run_end(energy, scf_converged, &format!("{scf_exit:?}"), extra);
    }
}

/// Print the two COSX energies of a final-grid pass (no-op without one).
fn print_cosx_final(result: &ferric_scf::result::ScfResult) {
    if let Some(f) = result.cosx_final {
        println!(
            "  COSX final grid: E(SCF grid, {} pts) = {:.10}, E(final grid, {} pts) = {:.10} \
             Hartree (reported; gradients use the SCF grid)",
            f.npts_scf, f.e_scf_grid, f.npts_final, f.e_final
        );
    }
}

/// The run-log record of a COSX final-grid pass: both energies, both point
/// counts, and which of the two an analytic gradient of this run
/// differentiates (the SCF-grid one; see `ferric_scf::cosx_k::CosxFinalPass`).
fn cosx_final_json(f: &ferric_scf::cosx_k::CosxFinalPass) -> serde_json::Value {
    serde_json::json!({
        "e_scf_grid": f.e_scf_grid,
        "e_final": f.e_final,
        "delta": f.e_final - f.e_scf_grid,
        "npts_scf_grid": f.npts_scf,
        "npts_final_grid": f.npts_final,
        "energy_reported": "e_final",
        "gradient_differentiates": "e_scf_grid",
    })
}

/// `RhfConfig::pcm` from `[pcm]` (`None` when the section is absent). The
/// values were already validated when the file was loaded, so the error arm
/// is a backstop. A named function rather than a closure in `run()`'s struct
/// literal, which is already at the project's complexity ceiling.
fn resolve_pcm(cfg: &Config) -> Option<ferric_pcm::PcmConfig> {
    match cfg.pcm.as_ref()?.to_pcm_config() {
        Ok(p) => Some(p),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

/// Resolve the molecule to solve, and the QM/MM system behind it if `[qmmm]`
/// is set.
///
/// Split out of `run` rather than inlined: `run`'s cyclomatic complexity is
/// already at the project ceiling, and the QM/MM branch added +8. The logic is
/// one decision -- QM region or plain molecule -- so it reads better alone.
fn resolve_geometry(cfg: &Config) -> (Option<ferric_scf::qmmm::QmmmSystem>, Molecule) {
    // A `[qmmm]` MM region and an explicit `[external_potential]` are two
    // sources for the same field; combining them would double a contribution
    // silently.
    if cfg.qmmm.is_some() && cfg.external_potential.to_external_potential().is_some() {
        eprintln!(
            "error: [qmmm] and [external_potential] both define an external field. \
             The QM/MM MM region IS an external potential, so combining them would \
             double a contribution silently. Remove one."
        );
        std::process::exit(1);
    }
    let Some(q) = cfg.qmmm.as_ref() else {
        let mol = Molecule::load_xyz_with_charge(
            &cfg.molecule.xyz,
            cfg.molecule.charge,
            cfg.molecule.multiplicity,
        )
        .unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });
        return (None, mol);
    };
    let sys = q
        .to_system(cfg.molecule.charge, cfg.molecule.multiplicity)
        .unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });
    let mol = sys.to_qm_molecule();
    eprintln!(
        "[ferric] QM/MM: {} QM atoms, {} MM charges from {}",
        mol.atoms.len(),
        sys.mm_charge_positions().len(),
        q.pqr
    );
    (Some(sys), mol)
}

/// Hill-notation molecular formula (C first, then H, then the rest
/// alphabetically), for the run log's molecule record.
///
/// A count of atoms alone does not identify a molecule; a formula plus the
/// input path does, well enough to tell two runs apart in a directory of logs.
fn molecular_formula(mol: &Molecule) -> String {
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for a in &mol.atoms {
        *counts.entry(a.symbol.as_str()).or_insert(0) += 1;
    }
    let mut out = String::new();
    let mut push = |sym: &str, n: usize| {
        out.push_str(sym);
        if n > 1 {
            out.push_str(&n.to_string());
        }
    };
    for sym in ["C", "H"] {
        if let Some(n) = counts.remove(sym) {
            push(sym, n);
        }
    }
    for (sym, n) in counts {
        push(sym, n);
    }
    out
}

/// `method.kind = "rhf"`. Extracted verbatim from the former `main()`
/// `"rhf" => { ... }` match arm.
fn run_rhf(
    cfg: &Config,
    bs: &BasisSet,
    prep: &PreparedBasis,
    result: &ferric_scf::result::ScfResult,
) {
    println!("RHF/{} on {}", bs.name, cfg.molecule.xyz);
    println!("  nbasis     = {}", prep.nbasis());
    println!("  iterations = {}", result.iterations);
    println!("  converged  = {}", result.converged);
    println!("  energy     = {:.10} Hartree", result.energy);
    print_cosx_final(result);
}

/// `method.kind = "ksdft"`. Extracted verbatim from the former `main()`
/// `"ksdft" => { ... }` match arm.
fn run_ksdft(
    cfg: &Config,
    bs: &BasisSet,
    prep: &PreparedBasis,
    result: &ferric_scf::result::ScfResult,
    dispersion: Option<&DispersionCorrection>,
) {
    let functional = cfg.dft.functional.as_deref().unwrap_or("LDA");
    println!("KS-DFT[{functional}]/{} on {}", bs.name, cfg.molecule.xyz);
    println!("  nbasis     = {}", prep.nbasis());
    println!("  iterations = {}", result.iterations);
    println!("  converged  = {}", result.converged);
    print_scf_energy(result.energy, dispersion);
    print_cosx_final(result);
}

/// A `[dft] dispersion` correction evaluated at one geometry.
///
/// Computed ONCE per single point and shared by the printout and the JSON run
/// log, so the two can never disagree and MBD's free-atom SCFs never run twice.
struct DispersionCorrection {
    /// Model label as printed and logged: `"D3(BJ)"` or `"MBD@rsSCS"`.
    model: &'static str,
    /// The functional whose published parameters were used.
    params: String,
    /// Dispersion energy (Hartree), ADDED to the SCF energy.
    energy: f64,
    /// MBD@rsSCS only: the range-separation β and the Hirshfeld volume ratios
    /// v_A / v_A^free the TS inputs were scaled by.
    mbd: Option<(f64, Vec<f64>)>,
}

impl DispersionCorrection {
    /// The dispersion lines of an SCF printout: the uncorrected KS energy, the
    /// correction, and the corrected total on the `energy` line.
    fn print(&self, scf_energy: f64) {
        println!("  E(KS-DFT)  = {scf_energy:.10} Hartree");
        match &self.mbd {
            None => {
                println!(
                    "  E(D3BJ)    = {:+.10} Hartree [params: {}]",
                    self.energy, self.params
                );
                println!(
                    "  energy     = {:.10} Hartree (KS-DFT + D3(BJ), two-body)",
                    scf_energy + self.energy
                );
            }
            Some((beta, _)) => {
                println!(
                    "  E(MBD@rsSCS) = {:+.10} Hartree [beta: {beta}, functional: {}]",
                    self.energy, self.params
                );
                println!(
                    "  energy     = {:.10} Hartree (KS-DFT + MBD@rsSCS)",
                    scf_energy + self.energy
                );
            }
        }
    }

    /// The `"dispersion"` object of the JSON run log.
    fn to_json(&self) -> serde_json::Value {
        let mut v = serde_json::json!({
            "model": self.model,
            "params": self.params,
            "energy": self.energy,
        });
        if let (Some((beta, ratios)), Some(obj)) = (&self.mbd, v.as_object_mut()) {
            obj.insert("beta".to_string(), serde_json::json!(beta));
            obj.insert("volume_ratios".to_string(), serde_json::json!(ratios));
        }
        v
    }
}

/// The parsed `[dft] dispersion` request, if the key is set. A parse error
/// (unknown spelling, or a functional with no published parameters) EXITS.
fn dispersion_request(cfg: &Config) -> Option<crate::config::DispersionRequest> {
    let spec = cfg.dft.dispersion.as_deref()?;
    Some(
        crate::config::DispersionRequest::parse_config_str(spec, cfg.dft.functional.as_deref())
            .unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            }),
    )
}

/// The `[dft] dispersion` correction at `mol`, if one was asked for, given the
/// converged SCF's spin-summed AO density (MBD@rsSCS takes its per-atom
/// polarizabilities from Hirshfeld volumes of it; D3(BJ) ignores it).
///
/// A failure EXITS rather than letting the caller print an uncorrected energy
/// under a heading that claims a correction was applied.
fn dispersion_correction(
    cfg: &Config,
    ctx: &ParallelContext,
    mol: &Molecule,
    bs: &BasisSet,
    op: Operator,
    rhf_config: &RhfConfig,
    density_total: &ndarray::Array2<f64>,
) -> Option<DispersionCorrection> {
    let req = dispersion_request(cfg)?;
    let evaluated = match req {
        crate::config::DispersionRequest::D3Bj { functional } => {
            ferric_d3::d3bj_params_for_functional(&functional)
                .and_then(|params| ferric_d3::d3bj_energy_for_molecule(mol, &params))
                .map(|e| DispersionCorrection {
                    model: "D3(BJ)",
                    params: functional,
                    energy: e,
                    mbd: None,
                })
        }
        crate::config::DispersionRequest::Mbd { functional } => {
            ferric_rpa::dispersion::mbd_rsscs::MbdRsscsConfig::for_functional(&functional).and_then(
                |mcfg| {
                    let cache = mbd_free_atom_cache(ctx, mol, bs, op, rhf_config);
                    let r = ferric_rpa::dispersion::mbd_scf::mbd_rsscs_for_density(
                        &cache,
                        mol,
                        bs,
                        density_total,
                        &mcfg,
                        false,
                    )?;
                    Ok(DispersionCorrection {
                        model: "MBD@rsSCS",
                        params: functional,
                        energy: r.energy,
                        mbd: Some((mcfg.beta, r.volume_ratios)),
                    })
                },
            )
        }
    };
    Some(evaluated.unwrap_or_else(|e| {
        eprintln!("error: [dft] dispersion: {e}");
        std::process::exit(1);
    }))
}

/// A `[dft] dispersion` model resolved ONCE for a task that needs its energy
/// AND gradient at many geometries (`optimize`, `frequencies`).
///
/// The energy and the gradient come from the SAME resolved parameters, which
/// is the property that makes optimizing on (or differentiating) this surface
/// meaningful -- a mismatched pair describes no surface at all.
enum DispersionGradientModel {
    /// D3(BJ) with `functional`'s damping parameters (geometry only).
    D3Bj {
        functional: String,
        params: ferric_d3::D3Params,
    },
    /// MBD@rsSCS: the free-atom references are per element, so they are built
    /// once here, not at every geometry.
    Mbd {
        functional: String,
        cache: ferric_rpa::dispersion::mbd_scf::MbdFreeAtomCache,
        mcfg: ferric_rpa::dispersion::mbd_rsscs::MbdRsscsConfig,
    },
}

impl DispersionGradientModel {
    /// The correction and its exact analytic nuclear gradient (natoms, 3) at
    /// `mol`, given the SCF converged there. For MBD@rsSCS the gradient
    /// includes the orbital relaxation of the Hirshfeld volumes (Z-vector).
    fn evaluate(
        &self,
        ctx: &ParallelContext,
        mol: &Molecule,
        bs: &BasisSet,
        op: Operator,
        rhf_config: &RhfConfig,
        scf: &ferric_scf::result::ScfResult,
    ) -> Result<(DispersionCorrection, ndarray::Array2<f64>), ferric_core::FerricError> {
        match self {
            DispersionGradientModel::D3Bj { functional, params } => {
                let e = ferric_d3::d3bj_energy_for_molecule(mol, params)?;
                let g = ferric_d3::d3bj_gradient_for_molecule(mol, params)?;
                let mut arr = ndarray::Array2::<f64>::zeros((g.len(), 3));
                for (k, row) in g.iter().enumerate() {
                    for a in 0..3 {
                        arr[[k, a]] = row[a];
                    }
                }
                let c = DispersionCorrection {
                    model: "D3(BJ)",
                    params: functional.clone(),
                    energy: e,
                    mbd: None,
                };
                Ok((c, arr))
            }
            DispersionGradientModel::Mbd {
                functional,
                cache,
                mcfg,
            } => {
                let r = ferric_rpa::dispersion::mbd_scf::mbd_rsscs_for_scf(
                    ctx, cache, mol, bs, op, rhf_config, scf, mcfg,
                )?;
                let g = r.gradient.ok_or_else(|| {
                    ferric_core::FerricError::General(
                        "MBD@rsSCS returned no gradient although one was requested".to_string(),
                    )
                })?;
                let c = DispersionCorrection {
                    model: "MBD@rsSCS",
                    params: functional.clone(),
                    energy: r.energy,
                    mbd: Some((mcfg.beta, r.volume_ratios)),
                };
                Ok((c, g))
            }
        }
    }
}

/// The `[dft] dispersion` model for a gradient-consuming `task`, if the key is
/// set. Every failure EXITS before any molecular SCF: an unknown functional,
/// or (MBD@rsSCS) a configuration whose exact gradient is unavailable, which
/// would otherwise fail after the first SCF.
fn dispersion_gradient_model(
    cfg: &Config,
    ctx: &ParallelContext,
    mol: &Molecule,
    bs: &BasisSet,
    op: Operator,
    rhf_config: &RhfConfig,
    task: &str,
    spin: ferric_scf::Spin,
) -> Option<DispersionGradientModel> {
    match dispersion_request(cfg)? {
        crate::config::DispersionRequest::D3Bj { functional } => {
            let params = ferric_d3::d3bj_params_for_functional(&functional).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            Some(DispersionGradientModel::D3Bj { functional, params })
        }
        crate::config::DispersionRequest::Mbd { functional } => {
            let mcfg =
                ferric_rpa::dispersion::mbd_rsscs::MbdRsscsConfig::for_functional(&functional)
                    .unwrap_or_else(|e| {
                        eprintln!("error: {e}");
                        std::process::exit(1);
                    });
            // The gradient's orbital-relaxation (Z-vector) term must be
            // available, or the run would fail after the first SCF.
            let unsupported = match spin {
                ferric_scf::Spin::Restricted => {
                    ferric_scf::zvector_ks::unsupported_reason(rhf_config)
                }
                ferric_scf::Spin::Unrestricted => {
                    ferric_scf::zvector_ks::unsupported_reason_unrestricted(rhf_config)
                }
                ferric_scf::Spin::RestrictedOpen => {
                    ferric_scf::zvector_ks::unsupported_reason_roks(rhf_config)
                }
            };
            if let Some(r) = unsupported {
                eprintln!(
                    "error: [dft] dispersion = \"mbd\" with method.task = \"{task}\": \
                     the exact MBD@rsSCS gradient is not available: {r}"
                );
                std::process::exit(1);
            }
            let cache = mbd_free_atom_cache(ctx, mol, bs, op, rhf_config);
            Some(DispersionGradientModel::Mbd {
                functional,
                cache,
                mcfg,
            })
        }
    }
}

/// The per-element free-atom data (Hirshfeld proatoms and live-SCF free-atom
/// volumes) MBD@rsSCS needs, built once per run. Its free-atom SCFs are
/// internal sub-solves, logged as such rather than as the run's SCF.
fn mbd_free_atom_cache(
    ctx: &ParallelContext,
    mol: &Molecule,
    bs: &BasisSet,
    op: Operator,
    rhf_config: &RhfConfig,
) -> ferric_rpa::dispersion::mbd_scf::MbdFreeAtomCache {
    let _sub = ferric_scf::runlog::SubSolveScope::enter();
    ferric_rpa::dispersion::mbd_scf::MbdFreeAtomCache::build(ctx, mol, bs, op, rhf_config)
        .unwrap_or_else(|e| {
            eprintln!("error: MBD@rsSCS free-atom reference failed: {e}");
            std::process::exit(1);
        })
}

/// The energy line(s) of an SCF printout (RKS, UKS, ROKS, or plain HF).
///
/// Dispersion is added only if `[dft] dispersion` asked for it, which `run()`
/// admits only on a Kohn-Sham SCF. With the key absent the output is
/// byte-identical to before the key existed: one "energy" line and no
/// dispersion line at all.
fn print_scf_energy(energy: f64, dispersion: Option<&DispersionCorrection>) {
    match dispersion {
        None => println!("  energy     = {energy:.10} Hartree"),
        Some(d) => d.print(energy),
    }
}

/// Runs that return before `run()`'s terminal `run_end` record (open-shell
/// UKS/ROKS energies, `task = "frequencies"`) log a dispersion correction as
/// its own `dispersion` record: the SCF energy, the corrected total, and the
/// model, at the run's input geometry.
fn log_scf_dispersion(scf_energy: f64, dispersion: Option<&DispersionCorrection>) {
    if let (Some(d), Some(rl)) = (dispersion, ferric_scf::runlog::log()) {
        rl.note(
            "dispersion",
            serde_json::json!({
                "scf_energy": scf_energy,
                "energy": scf_energy + d.energy,
                "dispersion": d.to_json(),
            }),
        );
    }
}

/// Print the `[ferric] SCF J/K:` line for an SCF about to run with `scf`.
///
/// `exchange_used` is the test every SCF solver applies before building any
/// K: plain HF, or a functional with nonzero short-range or range-separated
/// exact exchange. An unparseable name is reported by the SCF itself, so it
/// is treated as "uses K" here.
///
/// `open_shell`: `solve_uhf`/`solve_rohf` build J directly (four-centre) for
/// a range-separated functional whatever `df_j_aux` says (their `j_aux_eff`
/// requires omega == 0), so the line must say exact J there.
fn log_jk_path(scf: &RhfConfig, open_shell: bool) {
    let k_mix = scf
        .xc
        .as_deref()
        .and_then(|name| ferric_dft::libxc::xc_def_from_name(name).ok())
        .map(|d| ferric_dft::libxc::k_mix_from_xc_def(&d));
    let (exchange_used, rsh) = match &k_mix {
        None => (true, false),
        Some(m) => (m.sr > 0.0 || m.omega > 0.0, m.omega > 0.0),
    };
    let df_j = if open_shell && rsh {
        None
    } else {
        scf.df_j_aux.as_deref()
    };
    if scf.k_builder.as_deref() == Some("cosx") && exchange_used && !rsh {
        let j = df_j.filter(|s| !s.is_empty()).map_or_else(
            || "exact J (four-centre)".to_string(),
            |a| format!("RI-J via {a}"),
        );
        let g = &scf.cosx.grid;
        let prune = g
            .prune
            .map_or("flat".to_string(), |p| format!("{p:?}").to_lowercase());
        let fin = scf
            .cosx
            .final_grid
            .as_ref()
            .map_or("no final pass".to_string(), |f| {
                let fp = f
                    .prune
                    .map_or("flat".to_string(), |p| format!("{p:?}").to_lowercase());
                format!("final pass on ({}, {}, {fp})", f.n_radial, f.n_angular)
            });
        let tag = if j.starts_with("RI-J") {
            " (RIJCOSX)"
        } else {
            ""
        };
        eprintln!(
            "[ferric] SCF J/K: {j}, COSX K{tag}; COSX grid ({}, {}, {prune}), {fin}",
            g.n_radial, g.n_angular
        );
        return;
    }
    eprintln!(
        "[ferric] SCF J/K: {}",
        config::describe_jk_path(df_j, scf.df_k_aux.as_deref(), exchange_used)
    );
}

/// The UHF reference for an open-shell correlated kind: one that
/// `Config::validate_multiplicity` admitted (rimp2, oo-rimp2, pdep-rpa, gw,
/// mp2-v, all task = "energy").
///
/// pdep-rpa/gw/mp2-v keep the MOM-from-iteration-5 UHF they have always been
/// given here (pdep-rpa and gw re-solve their own UHF inside their arms;
/// mp2-v consumes this one, dispatching on `result.spin`). rimp2/oo-rimp2
/// get the plain UHF that `kind = "uhf"` runs with the same `[scf]` keys, so
/// the reference energy an open-shell RI-MP2 run prints IS the `uhf` energy.
#[allow(clippy::too_many_arguments)]
fn solve_open_shell_reference(
    method: &str,
    cfg: &Config,
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    rhf_config: &RhfConfig,
) -> ferric_scf::result::ScfResult {
    // Same config-honesty rule as the ladder path: these two are closed-shell
    // only, so say they were not used rather than ignore them silently.
    // Explicit settings only: df_guess defaults on, so df_guess_enabled()
    // would warn on every open-shell run.
    if cfg.scf.df_guess == Some(true) || cfg.scf.df_increments {
        eprintln!(
            "warning: [scf] df_guess / df_increments are closed-shell only; ignored for the \
             open-shell (UHF) reference of method.kind = \"{method}\""
        );
    }
    let mut uhf_cfg = rhf_config.clone();
    if matches!(method, "pdep-rpa" | "gw" | "mp2-v") {
        uhf_cfg.mom_after_iter = 5;
    }
    log_jk_path(&uhf_cfg, true);
    solve_uhf(ctx, mol, prep, bounds, &uhf_cfg).unwrap_or_else(|e| {
        eprintln!("error: UHF reference for method.kind = \"{method}\" failed: {e}");
        std::process::exit(1);
    })
}

/// The value, or print the error and exit 1 (the CLI's config-error path).
fn or_exit<T>(r: Result<T, String>) -> T {
    r.unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    })
}

/// A local (amplitude-threshold) run's model, as every printout and run-log
/// record of `rimp2`/`drpa`/`linlccd` states it.
struct LocalPrint {
    eps: f64,
    keep_fraction: f64,
    integral_direct: bool,
}

/// The first line of a `rimp2`/`drpa`/`linlccd` result block: the method
/// and its model, `"<method> (exact)"` or `"<method> (local: amplitude
/// threshold, eps = <eps>; kept <x>% of amplitudes)"`. The threshold is part
/// of the model, so a local number never prints without it.
fn model_label(method: &str, local: Option<&LocalPrint>) -> String {
    match local {
        None => format!("{method} (exact)"),
        Some(l) => format!(
            "{method} (local: amplitude threshold{}, eps = {:.1e}; kept {:.2}% of amplitudes)",
            if l.integral_direct {
                ", integral-direct"
            } else {
                ""
            },
            l.eps,
            100.0 * l.keep_fraction
        ),
    }
}

/// The run-log `local` component of a `rimp2`/`drpa`/`linlccd` result:
/// `null` for the exact method, else the scheme, threshold, kept fraction and
/// whether the integral-direct path ran.
fn local_json(local: Option<&LocalPrint>) -> serde_json::Value {
    match local {
        None => serde_json::Value::Null,
        Some(l) => serde_json::json!({
            "scheme": config::LocalScheme::AmplitudeThreshold.as_str(),
            "eps": l.eps,
            "keep_fraction": l.keep_fraction,
            "integral_direct": l.integral_direct,
        }),
    }
}

/// The exact-reference lines of a local run's printout.
///
/// `e_ref` is `None` when the (opt-in, `[local] reference = true`) reference
/// was not computed: the output then SAYS so, instead of printing the
/// library's NaN sentinel and a NaN difference. `ref_label` names the
/// reference line, `err_label` the difference line; `err_note` is its
/// parenthetical.
fn local_reference_lines(
    e_corr: f64,
    e_ref: Option<f64>,
    ref_label: &str,
    err_label: &str,
    err_note: &str,
) -> Vec<String> {
    match e_ref {
        Some(e_ref) => vec![
            format!("  {ref_label:<22}= {e_ref:.10} Ha"),
            format!("  {err_label:<22}= {:+.3e} Ha ({err_note})", e_corr - e_ref),
        ],
        None => vec![format!(
            "  {ref_label:<22}= not computed (opt-in: set [local] reference = true)"
        )],
    }
}

/// `method.kind = "rimp2"` with `[local] scheme = "amplitude-threshold"`:
/// amplitude-threshold local MP2 (`ferric_mp2::lmp2_amplitude`, WSHG23
/// single-threshold; closed-shell), or with `integral_direct = true` the
/// integral-direct local MP2 (`ferric_mp2::lmp2_direct`), which never forms
/// the global 3-index tensor and prints every locality map it used.
///
/// ε = 0 reproduces the exact RI-MP2 (library anchor <= 1e-9); a finite ε
/// carries a one-sided, ~linear-in-ε truncation error. The canonical RI-MP2
/// reference and the error against it are printed only with `[local]
/// reference = true` (OPT-IN: it is a full N^5 canonical RI-MP2 over the
/// global 3-index tensor -- the object the integral-direct path exists to
/// avoid). Measured record of the direct path: wiki/amplitude-threshold-lmp2.md.
#[allow(clippy::too_many_arguments)]
fn run_rimp2_local(
    cfg: &Config,
    model: &config::LocalModel,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    op: Operator,
    result: &ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
) {
    use ferric_mp2::lmp2_amplitude::{amplitude_lmp2, AmplitudeLmp2Config};
    use ferric_mp2::lmp2_direct::{amplitude_lmp2_direct, DirectConfig};
    require_restricted(result, "rimp2");
    let (aux_name, dfbs) = correlation_aux(cfg, mol);
    let eps = model.eps[0];
    let want_ref = model.reference;
    let lcfg = AmplitudeLmp2Config {
        eps,
        frozen_core: cfg.mp2.frozen_core.resolve(mol),
        eri3_budget_bytes: budget_bytes,
        compute_reference: want_ref,
        pair_gate_cal: model.direct.as_ref().and_then(|d| d.gate_cal),
        ..Default::default()
    };
    let (r, maps) = match &model.direct {
        None => (
            amplitude_lmp2(mol, prep, bs, &dfbs, op, result, &lcfg).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            }),
            None,
        ),
        Some(d) => {
            let dcfg = DirectConfig {
                aux_radius_bohr: d.aux_radius,
                virt_radius_bohr: Some(d.virt_radius),
                ao_tail: d.ao_tail,
                schwarz_skip: d.schwarz_skip,
                batch_merge: d.batch_merge,
                virt_schwarz_kappa: d.virt_schwarz_kappa,
                ..Default::default()
            };
            let (r, st) = amplitude_lmp2_direct(mol, prep, bs, &dfbs, op, result, &lcfg, &dcfg)
                .unwrap_or_else(|e| {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                });
            (r, Some((d, st)))
        }
    };
    let local = LocalPrint {
        eps,
        keep_fraction: r.keep_fraction,
        integral_direct: model.direct.is_some(),
    };
    println!("{}", model_label("MP2", Some(&local)));
    println!("  basis / aux           = {} / {aux_name}", bs.name);
    if let Some((d, _)) = &maps {
        println!(
            "  locality maps         = r_aux {} Bohr, r_virt {} Bohr, ao_tail {:.0e}, \
             schwarz_skip {:.0e}, batch_merge {}, gate_cal {}, virt_schwarz_kappa {}",
            d.aux_radius,
            d.virt_radius,
            d.ao_tail,
            d.schwarz_skip,
            d.batch_merge,
            d.gate_cal.map_or("off".to_string(), |c| format!("{c}")),
            d.virt_schwarz_kappa
                .map_or("off".to_string(), |k| format!("{k}")),
        );
    }
    println!("  RHF energy            = {:.10} Ha", result.energy);
    println!("  E_corr(local MP2)     = {:.10} Ha", r.e_corr);
    let e_ref = want_ref.then_some(r.e_corr_canonical_ri);
    let (err_label, err_note) = if maps.is_some() {
        ("total error", "eps truncation + locality maps")
    } else {
        ("threshold error", "one-sided; ~linear in eps")
    };
    for line in local_reference_lines(r.e_corr, e_ref, "E_corr(canonical RI)", err_label, err_note)
    {
        println!("{line}");
    }
    println!("  total energy          = {:.10} Ha", r.e_total);
    match &maps {
        None => println!(
            "  keep {:.4}  pairs {:.3}  dom(mean/max) {:.1}/{}  cg {}",
            r.keep_fraction, r.pair_fraction, r.dom_mean, r.dom_max, r.cg_iterations
        ),
        Some((_, st)) => {
            println!(
                "  keep {:.4}  pairs {:.3}  gated {}  dom(mean/max) {:.1}/{}  \
                 cand(mean/max) {:.1}/{}  cg {}",
                r.keep_fraction,
                r.pair_fraction,
                r.n_pairs_gated,
                r.dom_mean,
                r.dom_max,
                st.virt_cand_mean,
                st.virt_cand_max,
                r.cg_iterations
            );
            println!(
                "  strips rows {:.0}/{} cols {:.0}/{}  eri3 {:.1}M evald / {:.1}M skipped  \
                 t maps/eri3/metric/pairs/solve {:.2}/{:.2}/{:.2}/{:.2}/{:.2} s",
                st.strip_rows_mean,
                st.strip_rows_max,
                st.strip_cols_mean,
                st.strip_cols_max,
                st.n_eri3_shell_triples as f64 / 1e6,
                st.n_eri3_skipped as f64 / 1e6,
                st.t_maps_s,
                st.t_eri3_s,
                st.t_metric_s,
                st.t_pairs_s,
                r.timings.t_solve_s,
            );
        }
    }
    if let Some(rl) = ferric_scf::runlog::log() {
        rl.result(
            "rimp2",
            r.e_total,
            serde_json::json!({
                "e_corr": r.e_corr,
                // null when the opt-in reference was not computed
                "e_corr_canonical_ri": e_ref,
                "e_scf_reference": result.energy,
                "scf_converged": result.converged,
                "local": local_json(Some(&local)),
            }),
        );
    }
    if !result.converged {
        eprintln!(
            "warning: SCF did not converge (exit {:?} after {} iterations) — the correlation \
             energy above is built on an unconverged reference and must not be quoted",
            result.exit, result.iterations
        );
    }
}

/// `"rimp2" => { ... }` match arm.
fn run_rimp2(
    cfg: &Config,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    op: Operator,
    result: &ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
) {
    let model = cfg.local_model().unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    if let Some(model) = model {
        run_rimp2_local(cfg, &model, mol, bs, prep, op, result, budget_bytes);
        return;
    }
    // An open-shell molecule arrives with a UHF reference (see
    // `solve_open_shell_reference`) and takes the unrestricted RI-MP2.
    if result.spin != ferric_scf::result::Spin::Restricted {
        run_u_rimp2(cfg, mol, bs, prep, op, result, budget_bytes);
        return;
    }
    let aux_name = cfg
        .mp2
        .auxbasis
        .as_deref()
        .unwrap_or(config::DEFAULT_CORRELATION_AUX);
    let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let mp2_result = ri_mp2(
        mol,
        prep,
        &dfbs,
        op,
        result,
        &RiMp2Config {
            frozen_core: cfg.mp2.frozen_core.resolve(mol),
            memory_budget_bytes: budget_bytes,
            kappa: cfg.mp2.kappa,
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    println!("{}", model_label("RI-MP2", None));
    println!(
        "RI-MP2/{} (aux: {}) on {}",
        bs.name, aux_name, cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!("  RHF energy = {:.10} Hartree", result.energy);
    // SCF iteration count AND convergence state. Their previous absence was
    // actively harmful, not merely unhelpful: `solve_rhf` returns `Ok` on a
    // `MaxIter` exit carrying a best-effort density, so a run that never
    // converged printed an energy that LOOKED right (the density is close, so
    // the energy agrees to ~5e-10) with nothing to distinguish it from a
    // converged one. That hole produced two wrong performance diagnoses: an
    // integral-precision ramp measured as "6.5x slower" was in fact spinning to
    // max_iter — 100 iterations against the baseline's 12 — while reporting a
    // perfectly plausible energy. A wall time is uninterpretable unless the
    // iteration count is printed beside it.
    println!(
        "  SCF iters  = {}{}",
        result.iterations,
        if result.converged {
            ""
        } else {
            "  *** NOT CONVERGED ***"
        }
    );
    println!("  MP2 corr   = {:.10} Hartree", mp2_result.mp2_corr);
    println!("  Total      = {:.10} Hartree", mp2_result.total_energy);
    if let Some(rl) = ferric_scf::runlog::log() {
        // The ANSWER, not the SCF reference `run_end` carries. `scf_converged`
        // travels with it because an MP2 number built on an unconverged
        // reference must not be quoted -- the warning below says so on stderr,
        // and a machine reading the log needs the same signal.
        rl.result(
            "rimp2",
            mp2_result.total_energy,
            serde_json::json!({
                "e_corr": mp2_result.mp2_corr,
                "e_scf_reference": result.energy,
                "scf_converged": result.converged,
                "local": local_json(None),
            }),
        );
    }
    if !result.converged {
        eprintln!(
            "warning: SCF did not converge (exit {:?} after {} iterations) — the correlation \
             energy above is built on an unconverged reference and must not be quoted",
            result.exit, result.iterations
        );
    }
}

/// `method.kind = "rimp2"` on an open-shell molecule: unrestricted RI-MP2
/// (`ferric_mp2::u_rimp2::u_ri_mp2`, the UMP2 of PySCF's `mp.MP2(uhf)`) on
/// the UHF reference. Validated against PySCF UMP2 on OH/cc-pVDZ
/// (`u_rimp2_oh_cc_pvdz_matches_pyscf`) and against closed-shell RI-MP2 in
/// the closed-shell limit. (`u_ri_mp2` also accepts a ROHF reference, which it
/// semi-canonicalizes per spin; the CLI runs it on UHF.)
///
/// `[mp2] kappa` is refused: the regularizer is implemented for the
/// closed-shell kernel only, and silently dropping it would print plain UMP2
/// under a config asking for kappa-MP2.
fn run_u_rimp2(
    cfg: &Config,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    op: Operator,
    result: &ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
) {
    if let Some(k) = cfg.mp2.kappa {
        eprintln!(
            "error: [mp2] kappa = {k} is not supported for open-shell RI-MP2 (multiplicity = \
             {}): kappa-regularization exists for the closed-shell kernel only. Remove the key \
             for plain UMP2.",
            mol.multiplicity
        );
        std::process::exit(1);
    }
    let aux_name = cfg
        .mp2
        .auxbasis
        .as_deref()
        .unwrap_or(config::DEFAULT_CORRELATION_AUX);
    let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let r = ferric_mp2::u_rimp2::u_ri_mp2(
        mol,
        prep,
        &dfbs,
        op,
        result,
        &RiMp2Config {
            frozen_core: cfg.mp2.frozen_core.resolve(mol),
            memory_budget_bytes: budget_bytes,
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    // Same layout as the closed-shell printout (the "MP2 corr" / "Total"
    // labels are what downstream parsers read), with the reference named and
    // the three spin blocks shown.
    println!("{}", model_label("U-RI-MP2", None));
    println!(
        "U-RI-MP2/{} (aux: {}) on {}",
        bs.name, aux_name, cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!("  mult       = {}", mol.multiplicity);
    println!("  UHF energy = {:.10} Hartree", result.energy);
    println!(
        "  SCF iters  = {}{}",
        result.iterations,
        if result.converged {
            ""
        } else {
            "  *** NOT CONVERGED ***"
        }
    );
    println!("  E(aa)      = {:.10} Hartree", r.components.e_aa);
    println!("  E(bb)      = {:.10} Hartree", r.components.e_bb);
    println!("  E(ab)      = {:.10} Hartree", r.components.e_ab);
    println!("  MP2 corr   = {:.10} Hartree", r.mp2_corr);
    println!("  Total      = {:.10} Hartree", r.total_energy);
    if let Some(rl) = ferric_scf::runlog::log() {
        rl.result(
            "rimp2",
            r.total_energy,
            serde_json::json!({
                "reference": "UHF",
                "e_corr": r.mp2_corr,
                "e_aa": r.components.e_aa,
                "e_bb": r.components.e_bb,
                "e_ab": r.components.e_ab,
                "e_scf_reference": result.energy,
                "scf_converged": result.converged,
                "local": local_json(None),
            }),
        );
    }
    if !result.converged {
        eprintln!(
            "warning: SCF did not converge (exit {:?} after {} iterations) — the correlation \
             energy above is built on an unconverged reference and must not be quoted",
            result.exit, result.iterations
        );
    }
}

/// `method.kind = "mp3"`. Extracted verbatim from the former `main()`
/// `"mp3" => { ... }` match arm.
fn run_mp3(
    cfg: &Config,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    op: Operator,
    result: &ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
) {
    let aux_name = cfg
        .mp2
        .auxbasis
        .as_deref()
        .unwrap_or(config::DEFAULT_CORRELATION_AUX);
    let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let mp3_result = mp3_energy(
        mol,
        prep,
        &dfbs,
        op,
        result,
        cfg.mp2.frozen_core.resolve(mol),
        budget_bytes,
    )
    .unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    println!(
        "MP3/{} (aux: {}) on {}",
        bs.name, aux_name, cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!("  RHF energy = {:.10} Hartree", mp3_result.e_hf);
    println!("  MP2 corr   = {:.10} Hartree", mp3_result.e_mp2);
    println!("  MP3 corr   = {:.10} Hartree", mp3_result.e_mp3);
    println!("  Total corr = {:.10} Hartree", mp3_result.e_corr);
    println!("  Total      = {:.10} Hartree", mp3_result.e_total);
    if let Some(rl) = ferric_scf::runlog::log() {
        // MP3 carries BOTH orders separately, not just their sum: the MP2->MP3
        // step is the thing a reader checks for convergence of the series, and
        // a combined `e_corr` hides whether MP3 corrected or overcorrected.
        rl.result(
            "mp3",
            mp3_result.e_total,
            serde_json::json!({
                "e_corr": mp3_result.e_corr,
                "e_mp2": mp3_result.e_mp2,
                "e_mp3": mp3_result.e_mp3,
                "e_scf_reference": mp3_result.e_hf,
                "scf_converged": result.converged,
            }),
        );
    }
}

/// `method.kind = "oo-rimp2"`. Extracted verbatim from the former `main()`
/// `"oo-rimp2" => { ... }` match arm.
#[allow(clippy::too_many_arguments)]
fn run_oo_rimp2(
    cfg: &Config,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    result: &ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
    ext: Option<&ferric_core::external_potential::ExternalPotential>,
) {
    // An open-shell molecule arrives with a UHF reference (see
    // `solve_open_shell_reference`) and takes the unrestricted OO-RI-MP2.
    if result.spin != ferric_scf::result::Spin::Restricted {
        run_u_oo_rimp2(cfg, mol, bs, prep, op, bounds, result, budget_bytes, ext);
        return;
    }
    let aux_name = cfg
        .mp2
        .auxbasis
        .as_deref()
        .unwrap_or(config::DEFAULT_CORRELATION_AUX);
    let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    // `[mp2] oo_*` override the orbital-rotation loop; unset keys keep the
    // library default (the same values the Python `run_oo_rimp2` defaults to).
    let d = OoRiMp2Config::default();
    let oo_config = OoRiMp2Config {
        max_iter: cfg.mp2.oo_max_iter.unwrap_or(d.max_iter),
        grad_conv: cfg.mp2.oo_grad_conv.unwrap_or(d.grad_conv),
        level_shift: cfg.mp2.oo_level_shift.unwrap_or(d.level_shift),
        diis_size: cfg.mp2.oo_diis_size.unwrap_or(d.diis_size),
        frozen_core: cfg.mp2.frozen_core.resolve(mol),
        memory_budget_bytes: budget_bytes,
        verbose: cfg.scf.verbose,
        ..d
    };
    let oo_result = oo_ri_mp2(mol, prep, &dfbs, op, bounds, result, &oo_config, ext)
        .unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });
    println!(
        "OO-RI-MP2/{} (aux: {}) on {}",
        bs.name, aux_name, cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!("  converged  = {}", oo_result.converged);
    println!("  iterations = {}", oo_result.iterations);
    println!("  grad_norm  = {:.2e}", oo_result.grad_norm);
    println!("  HF energy  = {:.10} Hartree", oo_result.hf_energy);
    println!("  MP2 corr   = {:.10} Hartree", oo_result.mp2_corr);
    println!("  Total      = {:.10} Hartree", oo_result.total_energy);
    if let Some(rl) = ferric_scf::runlog::log() {
        rl.result(
            "oo-rimp2",
            oo_result.total_energy,
            serde_json::json!({
                "e_corr": oo_result.mp2_corr,
                "e_scf_reference": result.energy,
                "scf_converged": result.converged,
            }),
        );
    }
}

/// `method.kind = "oo-rimp2"` on an open-shell molecule: unrestricted
/// orbital-optimized RI-MP2 (`ferric_mp2::u_oo_rimp2::u_oo_ri_mp2`,
/// Bozkaya 2013) starting from the UHF reference. Its analytic orbital
/// gradient is validated against a PySCF finite-difference reference
/// (testdata/reference/oh_cc-pvdz_u-oomp2-fd.json), and it reduces to the
/// closed-shell OO-RI-MP2 on H2.
#[allow(clippy::too_many_arguments)]
fn run_u_oo_rimp2(
    cfg: &Config,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    result: &ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
    ext: Option<&ferric_core::external_potential::ExternalPotential>,
) {
    let aux_name = cfg
        .mp2
        .auxbasis
        .as_deref()
        .unwrap_or(config::DEFAULT_CORRELATION_AUX);
    let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    // Same `[mp2] oo_*` overrides as the closed-shell path.
    let d = ferric_mp2::u_oo_rimp2::UOoRiMp2Config::default();
    let oo_config = ferric_mp2::u_oo_rimp2::UOoRiMp2Config {
        max_iter: cfg.mp2.oo_max_iter.unwrap_or(d.max_iter),
        grad_conv: cfg.mp2.oo_grad_conv.unwrap_or(d.grad_conv),
        level_shift: cfg.mp2.oo_level_shift.unwrap_or(d.level_shift),
        diis_size: cfg.mp2.oo_diis_size.unwrap_or(d.diis_size),
        frozen_core: cfg.mp2.frozen_core.resolve(mol),
        memory_budget_bytes: budget_bytes,
        verbose: cfg.scf.verbose,
        ..d
    };
    let oo_result =
        ferric_mp2::u_oo_rimp2::u_oo_ri_mp2(mol, prep, &dfbs, op, bounds, result, &oo_config, ext)
            .unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
    // Same layout as the closed-shell printout; "HF energy" is the UHF
    // reference.
    println!(
        "U-OO-RI-MP2/{} (aux: {}) on {}",
        bs.name, aux_name, cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!("  mult       = {}", mol.multiplicity);
    println!("  converged  = {}", oo_result.converged);
    println!("  iterations = {}", oo_result.iterations);
    println!("  grad_norm  = {:.2e}", oo_result.grad_norm);
    println!("  HF energy  = {:.10} Hartree", oo_result.hf_energy);
    println!("  MP2 corr   = {:.10} Hartree", oo_result.mp2_corr);
    println!("  Total      = {:.10} Hartree", oo_result.total_energy);
    if let Some(rl) = ferric_scf::runlog::log() {
        rl.result(
            "oo-rimp2",
            oo_result.total_energy,
            serde_json::json!({
                "reference": "UHF",
                "e_corr": oo_result.mp2_corr,
                "e_scf_reference": result.energy,
                "scf_converged": result.converged,
                "oo_converged": oo_result.converged,
            }),
        );
    }
}

/// `method.kind = "att-rimp2"`. Extracted verbatim from the former `main()`
/// `"att-rimp2" => { ... }` match arm.
fn run_att_rimp2(
    cfg: &Config,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    result: &ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
) {
    let aux_name = cfg
        .mp2
        .auxbasis
        .as_deref()
        .unwrap_or(config::DEFAULT_CORRELATION_AUX);
    let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    // `[mp2] att_operator = "terfc"` selects the exact tempered erfc (Python
    // `run_terfc_rimp2`); the default erfc path below is unchanged.
    let att_op = cfg.mp2.att_rimp2_op().unwrap_or_else(|e| {
        eprintln!("config error: {e}");
        std::process::exit(1);
    });
    if att_op == config::AttRimp2Op::Terfc {
        run_att_rimp2_terfc(cfg, mol, bs, prep, &dfbs, aux_name, result, budget_bytes);
        return;
    }
    let omega_ang_inv = cfg.mp2.omega.unwrap_or(0.420);
    let att_config = AttenuatedMp2Config {
        omega: omega_ang_inv * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV,
        scaling: 1.0,
        frozen_core: cfg.mp2.frozen_core.resolve(mol),
        screen_thresh: None,
        memory_budget_bytes: budget_bytes,
    };
    let att_result = attenuated_ri_mp2(mol, prep, &dfbs, result, &att_config).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    // Name the operator explicitly. `attenuated_ri_mp2` is erfc-only
    // (attenuated.rs: `Operator::erfc(config.omega)`), while `scs-mp2-2terfc`
    // and `mp2-v` use terfc. An output that says only "Attenuated RI-MP2"
    // cannot be told apart from a terfc run downstream -- and a hardcoded
    // operator label in the rs-mp2-rpa arm caused exactly that confusion
    // (fixed 2026-07-26).
    println!(
        "Attenuated RI-MP2 (erfc)/{} (aux: {}, ω={:.3} Å⁻¹) on {}",
        bs.name, aux_name, omega_ang_inv, cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!("  RHF energy = {:.10} Hartree", result.energy);
    println!("  MP2 corr   = {:.10} Hartree", att_result.mp2_corr);
    println!(
        "  E_OS       = {:.10} Hartree",
        att_result.spin_components.e_os
    );
    println!(
        "  E_SS       = {:.10} Hartree",
        att_result.spin_components.e_ss
    );
    println!("  Total      = {:.10} Hartree", att_result.total_energy);
    if let Some(rl) = ferric_scf::runlog::log() {
        rl.result(
            "att-rimp2",
            att_result.total_energy,
            serde_json::json!({
                "e_corr": att_result.mp2_corr,
                "e_scf_reference": result.energy,
                "scf_converged": result.converged,
            }),
        );
    }
}

/// `method.kind = "att-rimp2"` with `[mp2] att_operator = "terfc"`: RI-MP2 with
/// the EXACT tempered-erfc operator `terfc(r, r0)/r` (Dutoi/Goldey
/// interpolation tables, `FERRIC_TERF_TABLE_DIR`) at `r0 = [mp2] att_r0` Å
/// (default 1.05). The SCF stays full Coulomb; only the correlation is
/// attenuated. The same call as the Python `run_terfc_rimp2`:
/// `ri_mp2(.., Operator::terfc(r0_bohr), ..)` with the CLI's frozen core and
/// memory budget.
#[allow(clippy::too_many_arguments)]
fn run_att_rimp2_terfc(
    cfg: &Config,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    dfbs: &PreparedBasis,
    aux_name: &str,
    result: &ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
) {
    const ANG2BOHR_R0: f64 = 1.8897259886;
    let r0_ang = cfg
        .mp2
        .att_r0
        .unwrap_or(config::ATT_RIMP2_TERFC_DEFAULT_R0_ANG);
    let mp2_config = RiMp2Config {
        frozen_core: cfg.mp2.frozen_core.resolve(mol),
        memory_budget_bytes: budget_bytes,
        ..Default::default()
    };
    let (sc, _) = ferric_mp2::rimp2::ri_mp2_spin_components(
        mol,
        prep,
        dfbs,
        Operator::terfc(r0_ang * ANG2BOHR_R0),
        result,
        &mp2_config,
    )
    .unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let total = result.energy + sc.e_total;
    println!(
        "Attenuated RI-MP2 (terfc)/{} (aux: {}, r0={:.3} Å) on {}",
        bs.name, aux_name, r0_ang, cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!("  RHF energy = {:.10} Hartree", result.energy);
    println!("  MP2 corr   = {:.10} Hartree", sc.e_total);
    println!("  E_OS       = {:.10} Hartree", sc.e_os);
    println!("  E_SS       = {:.10} Hartree", sc.e_ss);
    println!("  Total      = {:.10} Hartree", total);
    if let Some(rl) = ferric_scf::runlog::log() {
        rl.result(
            "att-rimp2",
            total,
            serde_json::json!({
                "operator": "terfc",
                "r0_angstrom": r0_ang,
                "e_corr": sc.e_total,
                "e_scf_reference": result.energy,
                "scf_converged": result.converged,
            }),
        );
    }
}

/// `method.kind = "rs-mp2-rpa"`. Extracted verbatim from the former `main()`
/// `"rs-mp2-rpa" => { ... }` match arm.
fn run_rs_mp2_rpa(
    cfg: &Config,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    result: &ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
) {
    let aux_name = cfg
        .mp2
        .auxbasis
        .as_deref()
        .unwrap_or(config::DEFAULT_CORRELATION_AUX);
    let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let omega_ang_inv = cfg.mp2.omega.unwrap_or(0.420);
    let formulation = match cfg.mp2.formulation.as_deref().unwrap_or("delta-lr") {
        "delta-lr" => ferric_rpa::RsMp2RpaFormulation::DeltaLr,
        "coupled-rings" => ferric_rpa::RsMp2RpaFormulation::CoupledRings,
        other => {
            eprintln!("error: unknown [mp2] formulation = \"{other}\"; expected \"delta-lr\" or \"coupled-rings\"");
            std::process::exit(1);
        }
    };
    let attenuator = match cfg.mp2.attenuator.as_deref().unwrap_or("erf") {
        "erf" => ferric_rpa::rs_mp2_rpa::Attenuator::Erf,
        "terf" => ferric_rpa::rs_mp2_rpa::Attenuator::Terf,
        other => {
            eprintln!(
                "error: unknown [mp2] attenuator = \"{other}\"; expected \"erf\" or \"terf\""
            );
            std::process::exit(1);
        }
    };
    // [mp2] r0 is Å at the CLI boundary (2026-07-21: fixed from Bohr, matching
    // r0_bonded/r0_nonbonded's existing Å convention below); only meaningful
    // for terf. Default matches the erf operating point (r0=1.6828 Å = 3.18
    // Bohr ⇒ ω≈0.42 Å⁻¹). Converted to Bohr immediately for RsMp2RpaConfig,
    // which stays Bohr-native (Operator::terf/terfc, the FFI shim, and the
    // terf-tables interpolation grids are all hard-Bohr all the way down).
    const ANG2BOHR_R0: f64 = 1.8897259886;
    let r0_ang = cfg.mp2.r0.unwrap_or(3.18 / ANG2BOHR_R0);
    let r0 = r0_ang * ANG2BOHR_R0;
    if matches!(attenuator, ferric_rpa::rs_mp2_rpa::Attenuator::Terf) && cfg.mp2.omega.is_some() {
        eprintln!("warning: [mp2] omega is ignored when attenuator = \"terf\" (ω is derived from r0 = {r0_ang} Å = {r0:.4} Bohr as ω = 1/(r0·√2))");
    }

    // [mp2] r0_sweep: evaluate several r0 in one job, reusing the SCF above.
    // The SCF is already done by the time we get here, so each extra point
    // costs only the correlation stage — that is the whole point (an N-point
    // scan for ~1 SCF instead of N).
    let r0_sweep: Option<Vec<f64>> = cfg.mp2.r0_sweep.as_ref().map(|v| {
        let mut s: Vec<f64> = v.clone();
        // NaN must NOT panic here. TOML accepts the `nan` literal, so this is
        // reachable from user config — and the finiteness check below is
        // written precisely to reject it with an actionable message. An
        // `.expect()` here fired FIRST and turned that clean diagnostic into a
        // raw backtrace, defeating the validation. Sort NaN-tolerantly and let
        // the real check do its job.
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        s.dedup();
        s
    });
    if let Some(s) = &r0_sweep {
        if s.is_empty() {
            eprintln!("error: [mp2] r0_sweep is empty");
            std::process::exit(1);
        }
        if s.iter().any(|&x| !(x > 0.0) || !x.is_finite()) {
            eprintln!("error: [mp2] r0_sweep values must be finite and > 0 (got {s:?})");
            std::process::exit(1);
        }
        if !matches!(attenuator, ferric_rpa::rs_mp2_rpa::Attenuator::Terf) {
            eprintln!(
                "error: [mp2] r0_sweep requires attenuator = \"terf\" (r0 is meaningless for erf)"
            );
            std::process::exit(1);
        }
        if cfg.mp2.r0.is_some() {
            eprintln!("warning: [mp2] r0 is ignored when r0_sweep is set");
        }
    }
    let mut rs_cfg = ferric_rpa::rs_mp2_rpa::RsMp2RpaConfig {
        omega: omega_ang_inv * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV,
        attenuator,
        r0,
        // `[mp2] terf_omega` (Å⁻¹, terf only -- validated by
        // `Config::validate_cli_wired_keys`); None keeps the curvature link.
        terf_omega: cfg
            .mp2
            .terf_omega
            .map(|w| w * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV),
        frozen_core: cfg.mp2.frozen_core.resolve(mol),
        formulation,
        ..Default::default()
    };
    // [rpa] trunc_thresh opts into PDEP truncation for the dRPA solves
    // (default 0.0 = full rank; production-size opt-in, validate vs
    // full-rank per system class before trusting).
    if let Some(t) = cfg.rpa.trunc_thresh {
        rs_cfg.drpa.trunc_thresh = t;
    }
    rs_cfg.drpa.memory_budget_bytes = budget_bytes;

    // One point per r0 in the sweep (or just the single configured r0). The
    // SCF `result` is shared across all of them by construction.
    let points: Vec<f64> = r0_sweep.clone().unwrap_or_else(|| vec![r0_ang]);
    let n_points = points.len();
    for (k, r0_ang_k) in points.into_iter().enumerate() {
        rs_cfg.r0 = r0_ang_k * ANG2BOHR_R0;
        if n_points > 1 {
            println!(
                "\n===== r0 sweep point {}/{}: r0 = {:.4} Å =====",
                k + 1,
                n_points,
                r0_ang_k
            );
        }
        emit_rs_mp2_rpa_point(
            cfg,
            mol,
            bs,
            prep,
            &dfbs,
            aux_name,
            result,
            &rs_cfg,
            omega_ang_inv,
            r0_ang_k,
        );
    }
}

/// Solve and print ONE `rs-mp2-rpa` point at the r0 already set in `rs_cfg`.
///
/// Split out of [`run_rs_mp2_rpa`] so `[mp2] r0_sweep` can call it once per r0
/// against a single converged SCF. The printed block is byte-identical to the
/// single-point output, so existing parsers (`benchmarks/grid/*.py`) keep
/// working on both layouts.
#[allow(clippy::too_many_arguments)]
fn emit_rs_mp2_rpa_point(
    cfg: &Config,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    dfbs: &PreparedBasis,
    aux_name: &str,
    result: &ferric_scf::result::ScfResult,
    rs_cfg: &ferric_rpa::rs_mp2_rpa::RsMp2RpaConfig,
    omega_ang_inv: f64,
    r0_ang: f64,
) {
    let r = ferric_rpa::rs_mp2_rpa::rs_mp2_lr_rpa(mol, prep, dfbs, result, rs_cfg).unwrap_or_else(
        |e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        },
    );
    println!(
        "RS-MP2-RPA/{} (aux: {}, ω={:.3} Å⁻¹) on {}",
        bs.name, aux_name, omega_ang_inv, cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    match rs_cfg.attenuator {
        ferric_rpa::rs_mp2_rpa::Attenuator::Erf => {
            println!(
                "RS-MP2-RPA [erf split] (ω = {omega_ang_inv:.3} Å⁻¹ = {:.4} Bohr⁻¹)",
                rs_cfg.omega
            );
        }
        ferric_rpa::rs_mp2_rpa::Attenuator::Terf => match rs_cfg.terf_omega {
            None => {
                let w_derived = 1.0 / (rs_cfg.r0 * std::f64::consts::SQRT_2);
                println!("RS-MP2-RPA [terf split] (r0 = {r0_ang:.4} Å = {:.4} Bohr, ω = 1/(r0·√2) = {:.4} Bohr⁻¹)", rs_cfg.r0, w_derived);
            }
            // Print the ω actually used: with `[mp2] terf_omega` the
            // curvature link is broken, so the derived value would be wrong.
            Some(w) => {
                println!("RS-MP2-RPA [terf split] (r0 = {r0_ang:.4} Å = {:.4} Bohr, ω = {:.4} Bohr⁻¹ from [mp2] terf_omega, decoupled from r0)", rs_cfg.r0, w);
            }
        },
    }
    // Common lines printed for all formulations.
    //
    // The SR/LR operator names MUST follow the attenuator actually in use.
    // These were hardcoded "erfc"/"erf" and so were WRONG for every terf-split
    // run: with `attenuator = "terf"` the operators are terf/terfc (see
    // rs_mp2_rpa.rs, `Attenuator::Terf => (Operator::terf, Operator::terfc)`).
    // A mislabelled component is worse than an unlabelled one -- it was read
    // downstream as erfc-attenuated MP2 when it is terfc-attenuated.
    let (sr_name, lr_name) = match rs_cfg.attenuator {
        ferric_rpa::rs_mp2_rpa::Attenuator::Erf => ("erfc", "erf"),
        ferric_rpa::rs_mp2_rpa::Attenuator::Terf => ("terfc", "terf"),
    };
    println!("  E(MP2, Coulomb)      = {:>16.10} Hartree", r.e_mp2_full);
    println!(
        "  {:<20} = {:>16.10} Hartree",
        format!("E(SR-MP2, {sr_name})"),
        r.e_sr_mp2
    );
    println!(
        "  {:<20} = {:>16.10} Hartree",
        format!("E(LR-MP2, {lr_name})"),
        r.e_lr_mp2
    );
    println!(
        "  {:<20} = {:>16.10} Hartree",
        format!("E(dMP2, {lr_name})"),
        r.e_dmp2_lr
    );
    // Formulation-specific lines.
    match rs_cfg.formulation {
        ferric_rpa::RsMp2RpaFormulation::DeltaLr => {
            println!(
                "  {:<20} = {:>16.10} Hartree",
                format!("E(dRPA, {lr_name})"),
                r.e_drpa_lr.unwrap()
            );
            println!("  E_corr naive (A)     = {:>16.10} Hartree   [diagnostic: misses SR×LR cross terms]", r.e_corr_naive.unwrap());
            println!("  E_corr Δ-form (B)    = {:>16.10} Hartree", r.e_corr);
        }
        ferric_rpa::RsMp2RpaFormulation::CoupledRings => {
            println!(
                "  E(ΔdRPA, Coulomb)    = {:>16.10} Hartree",
                r.e_delta_drpa_full.unwrap()
            );
            println!(
                "  {:<20} = {:>16.10} Hartree",
                format!("E(ΔdRPA, {sr_name})"),
                r.e_delta_drpa_sr.unwrap()
            );
            println!("  E_corr coupled (T)   = {:>16.10} Hartree", r.e_corr);
        }
    }
    println!("  Total energy         = {:>16.10} Hartree", r.total_energy);
    if let Some(rl) = ferric_scf::runlog::log() {
        rl.result(
            "rs-mp2-rpa",
            r.total_energy,
            serde_json::json!({
                "e_mp2_full": r.e_mp2_full,
                "e_sr_mp2": r.e_sr_mp2,
                "e_lr_mp2": r.e_lr_mp2,
                "e_scf_reference": result.energy,
                "scf_converged": result.converged,
            }),
        );
    }
}

/// `method.kind = "scs-mp2"`. Extracted verbatim from the former `main()`
/// `"scs-mp2" => { ... }` match arm.
fn run_scs_mp2(
    cfg: &Config,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    result: &ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
) {
    let aux_name = cfg
        .mp2
        .auxbasis
        .as_deref()
        .unwrap_or(config::DEFAULT_CORRELATION_AUX);
    let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let scs_config = ScsMp2Config {
        c_os: cfg.mp2.c_os.unwrap_or(6.0 / 5.0),
        c_ss: cfg.mp2.c_ss.unwrap_or(1.0 / 3.0),
        frozen_core: cfg.mp2.frozen_core.resolve(mol),
        memory_budget_bytes: budget_bytes,
    };
    let scs_result = scs_mp2(mol, prep, &dfbs, result, &scs_config).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    println!(
        "SCS-MP2/{} (aux: {}, c_OS={:.3}, c_SS={:.3}) on {}",
        bs.name, aux_name, scs_config.c_os, scs_config.c_ss, cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!("  RHF energy = {:.10} Hartree", result.energy);
    println!("  SCS corr   = {:.10} Hartree", scs_result.scs_corr);
    println!("  E_OS       = {:.10} Hartree", scs_result.e_os);
    println!("  E_SS       = {:.10} Hartree", scs_result.e_ss);
    println!("  Total      = {:.10} Hartree", scs_result.total_energy);
    if let Some(rl) = ferric_scf::runlog::log() {
        rl.result(
            "scs-mp2",
            scs_result.total_energy,
            serde_json::json!({
                "e_corr": scs_result.scs_corr,
                "e_os": scs_result.e_os,
                "e_ss": scs_result.e_ss,
                "e_scf_reference": result.energy,
                "scf_converged": result.converged,
            }),
        );
    }
}

/// `method.kind = "scs-mp2-2terfc"`. Extracted verbatim from the former
/// `main()` `"scs-mp2-2terfc" => { ... }` match arm.
fn run_scs_mp2_2terfc(
    cfg: &Config,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    result: &ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
) {
    let aux_name = cfg
        .mp2
        .auxbasis
        .as_deref()
        .unwrap_or(config::DEFAULT_CORRELATION_AUX);
    let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    // r0(1)/r0(2) are given in Å in the TOML (matching the Python
    // binding's convention); the library config wants Bohr.
    const ANG2BOHR: f64 = 1.8897259886;
    let r0_bonded_ang = cfg.mp2.r0_bonded.unwrap_or(0.75);
    let r0_nonbonded_ang = cfg.mp2.r0_nonbonded.unwrap_or(1.05);
    let scs_config = ScsMp2TerfcConfig {
        r0_bonded: r0_bonded_ang * ANG2BOHR,
        r0_nonbonded: r0_nonbonded_ang * ANG2BOHR,
        c_os: cfg.mp2.c_os.unwrap_or(1.27),
        c_ss: cfg.mp2.c_ss.unwrap_or(4.05),
        frozen_core: cfg.mp2.frozen_core.resolve(mol),
        memory_budget_bytes: budget_bytes,
    };
    if scs_config.r0_nonbonded <= scs_config.r0_bonded {
        eprintln!("error: [mp2] r0_nonbonded must be > r0_bonded");
        std::process::exit(1);
    }
    let scs_result = scs_mp2_2terfc(mol, prep, &dfbs, result, &scs_config).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    println!(
        "SCS-MP2(2terfc)/{} (aux: {}, r0(1)={:.3} Å, r0(2)={:.3} Å, c_OS={:.3}, c_SS={:.3}) on {}",
        bs.name,
        aux_name,
        r0_bonded_ang,
        r0_nonbonded_ang,
        scs_config.c_os,
        scs_config.c_ss,
        cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!("  RHF energy = {:.10} Hartree", result.energy);
    println!("  SCS corr   = {:.10} Hartree", scs_result.scs_corr);
    println!("  E_OS       = {:.10} Hartree", scs_result.e_os);
    println!("  E_SS       = {:.10} Hartree", scs_result.e_ss);
    println!("  Total      = {:.10} Hartree", scs_result.total_energy);
    if let Some(rl) = ferric_scf::runlog::log() {
        rl.result(
            "scs-mp2-2terfc",
            scs_result.total_energy,
            serde_json::json!({
                "e_corr": scs_result.scs_corr,
                "e_os": scs_result.e_os,
                "e_ss": scs_result.e_ss,
                "e_scf_reference": result.energy,
                "scf_converged": result.converged,
            }),
        );
    }
}

/// `method.kind = "mp2-v"`: attenuated MP2 + long-range VV10 dispersion
/// ("MP2-V", Goldey/Belzunces/Head-Gordon, JCTC 11, 4159 (2015)).
///
/// Structured after `run_att_rimp2`/`run_scs_mp2_2terfc` (same aux-basis
/// resolution, same Å→Bohr boundary, same print block) with two additions the
/// library API forces:
///
///  * MP2-V's VV10 half needs the **unprepared** `BasisSet` (`obs_bs`) on top
///    of the `PreparedBasis`, because it evaluates AOs and their gradients on a
///    real-space grid (`ferric_dft::ao_grid::eval_basis_and_grad_on_points`),
///    which the shell-list form carries and `PreparedBasis` does not. `bs` was
///    already threaded to every arm, so this costs nothing.
///  * Spin dispatch, following the `run_ccsd` precedent exactly: branch on
///    `result.spin` and call the matching library entry point. The open-shell
///    reference itself comes from `main()`'s pre-arm SCF, whose UHF+MOM
///    fallback `mp2-v` opts into alongside `pdep-rpa`/`gw` (the shared
///    `solve_rhf` ignores multiplicity and fails outright on an odd-electron
///    molecule). So `result` is already UHF here for `multiplicity > 1`; this
///    function never re-solves.
fn run_mp2_v(
    cfg: &Config,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    result: &ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
) {
    let aux_name = cfg
        .mp2
        .auxbasis
        .as_deref()
        .unwrap_or(config::DEFAULT_CORRELATION_AUX);
    let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let att_cfg = cfg
        .mp2
        .build_att_vv10_config(mol, budget_bytes)
        .unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });

    // Dispatch on the reference's spin, exactly as `run_ccsd` does. The two
    // library entry points reject the wrong spin (a restricted result routed
    // through the unrestricted path would silently take the alpha orbitals as
    // an independent spin channel), so this branch is a correctness gate, not
    // an optimization.
    let is_closed_shell = matches!(result.spin, ferric_scf::result::Spin::Restricted);
    // Padded to the same width as the other row labels below ("attMP2corr").
    let ref_label = if is_closed_shell {
        "RHF energy "
    } else {
        "SCF energy "
    };
    let mp2v = if is_closed_shell {
        att_mp2_vv10(mol, prep, bs, &dfbs, result, &att_cfg)
    } else {
        u_att_mp2_vv10(mol, prep, bs, &dfbs, result, &att_cfg)
    }
    .unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });

    // The library flags this itself; surface it rather than let a user read an
    // open-shell number as if the parameters had been fitted for it.
    if mp2v.is_open_shell_extrapolation() {
        eprintln!(
            "[warning] MP2-V on an open-shell reference: (r0, b, C) were fitted on S66, which is \
             entirely CLOSED-SHELL dimers. There is no published open-shell MP2-V \
             parameterization -- this is unparameterized extrapolation."
        );
    }

    let attenuator = match att_cfg.attenuator {
        AttVv10Attenuator::Terfc => "terfc",
        AttVv10Attenuator::Erfc => "erfc",
    };
    let damping = match att_cfg.vv10_damping {
        ferric_dft::vv10::Vv10Damping::Terfc { .. } => "terfc",
        ferric_dft::vv10::Vv10Damping::None => "none",
    };
    // Decoupled seam sharpness (2026-08): echo it in the boundary unit (Å⁻¹)
    // so a sweep's output is self-describing; the linked default prints
    // nothing extra (byte-identical behavior, no new label to misread).
    let omega_note = match att_cfg.omega {
        Some(w_bohr_inv) => format!(
            ", omega={:.4} Å⁻¹ (decoupled; UNPARAMETERIZED — b was fitted at the linked width)",
            w_bohr_inv / ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV
        ),
        None => String::new(),
    };
    println!(
        "MP2-V({attenuator})/{} (aux: {}, r0={:.3} Å{omega_note}, b={:.3}, C={:.4}, VV10 damping: {damping}) on {}",
        bs.name,
        aux_name,
        att_cfg.r0_angstrom(),
        att_cfg.vv10.b,
        att_cfg.vv10.c,
        cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!("  {ref_label}= {:.10} Hartree", mp2v.e_hf);
    println!("  attMP2corr = {:.10} Hartree", mp2v.e_c_att_mp2);
    match &mp2v.spin_components {
        AttVv10SpinComponents::Restricted(s) => {
            println!("  E_OS       = {:.10} Hartree", s.e_os);
            println!("  E_SS       = {:.10} Hartree", s.e_ss);
        }
        AttVv10SpinComponents::Unrestricted(u) => {
            println!("  E_aa       = {:.10} Hartree", u.e_aa);
            println!("  E_bb       = {:.10} Hartree", u.e_bb);
            println!("  E_ab       = {:.10} Hartree", u.e_ab);
        }
    }
    println!("  VV10 E_nl  = {:.10} Hartree", mp2v.e_nl_vv10);
    println!("  NLC grid   = {} points", mp2v.n_nlc_points);
    println!("  Total      = {:.10} Hartree", mp2v.total);
    if let Some(rl) = ferric_scf::runlog::log() {
        rl.result(
            "mp2-v",
            mp2v.total,
            serde_json::json!({
                "e_corr": mp2v.total - result.energy,
                "e_scf_reference": result.energy,
                "scf_converged": result.converged,
            }),
        );
    }
}

/// `method.kind = "ccsd"`. Extracted verbatim from the former `main()`
/// `"ccsd" => { ... }` match arm.
fn run_ccsd(
    cfg: &Config,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    op: Operator,
    result: &ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
) {
    let aux_name = cfg
        .mp2
        .auxbasis
        .as_deref()
        .unwrap_or(config::DEFAULT_CORRELATION_AUX);
    let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let cc_config = CcConfig {
        frozen_core: cfg.mp2.frozen_core.resolve(mol),
        memory_budget_bytes: budget_bytes,
        ..Default::default()
    };
    // Dispatch on the reference's spin. Both solvers compute the SAME CCSD
    // energy, but the spin-adapted one works in spatial orbitals (no/nv) rather
    // than spin orbitals (2no/2nv), so its O(N^6) VVVV block is 16x smaller —
    // measured ~8-10x faster at cc-pVDZ, and it is the algorithm PySCF's
    // `cc.CCSD` uses. Routing every closed-shell job through the spin-orbital
    // path was leaving that on the floor: water/aug-cc-pVDZ was 24.7 s here vs
    // 1.1 s for PySCF RCCSD, and only ~2.5x of that was implementation.
    //
    // `ccsd_closed_shell` requires a restricted reference (it calls `eps_r()`/
    // `mos_r()`, which assert on `Spin::Restricted`) — but so does the
    // spin-orbital `ccsd`, so this is a strict upgrade, not a narrowing. The
    // fallback exists so a future UHF/ROHF-fed CCSD keeps working rather than
    // silently taking a path that assumes closed shells.
    let is_closed_shell = matches!(result.spin, ferric_scf::result::Spin::Restricted);
    let solver: &str = if is_closed_shell {
        "spin-adapted"
    } else {
        "spin-orbital"
    };
    let cc_result = if is_closed_shell {
        ccsd_closed_shell(mol, prep, &dfbs, op, result, &cc_config)
    } else {
        ccsd(mol, prep, &dfbs, op, result, &cc_config)
    }
    .unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    println!(
        "CCSD/{} (aux: {}, {solver}) on {}",
        bs.name, aux_name, cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!("  RHF energy = {:.10} Hartree", result.energy);
    println!(
        "  CCSD corr  = {:.10} Hartree",
        cc_result.correlation_energy
    );
    println!(
        "  Total      = {:.10} Hartree",
        result.energy + cc_result.correlation_energy
    );
    if let Some(rl) = ferric_scf::runlog::log() {
        rl.result(
            "ccsd",
            result.energy + cc_result.correlation_energy,
            serde_json::json!({
                "e_corr": cc_result.correlation_energy,
                "e_scf_reference": result.energy,
                "scf_converged": result.converged,
            }),
        );
    }
}

/// `method.kind = "linlccd"`. Linearized ladder CCD on the converged
/// closed-shell reference, in the ladder variant `[mp2] linlccd_variant`
/// (`hh` default = LinLCCD(hh); `drivers-only` reproduces RI-MP2; `full`
/// adds the pp ladder with CCD-like VVVV memory).
///
/// Exact by default: the canonical `ferric_cc::linlccd::linlccd`, which
/// supports all three variants. With `[local] scheme = "amplitude-threshold"`
/// it is the amplitude-threshold LinLCCD in the localized basis
/// (`ferric_cc::linlccd_amplitude`); `eps = 0` reproduces the exact method of
/// the same variant. Aux basis as `run_ccsd` (`[mp2] auxbasis`, default
/// `cc-pvdz-ri`).
fn run_linlccd(
    cfg: &Config,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    op: Operator,
    result: &ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
) {
    // `linlccd` is closed-shell (RHF-reference) only — it calls `eps_r()`/`mos_r()`,
    // which assert on `Spin::Restricted`. Reject an open-shell reference here with a
    // clear message instead of letting that assert fire as a panic.
    if !matches!(result.spin, ferric_scf::result::Spin::Restricted) {
        eprintln!(
            "error: method.kind = \"linlccd\" requires a closed-shell (RHF) reference; \
             open-shell LinLCCD is library-only (ferric_cc::linlccd_u)"
        );
        std::process::exit(1);
    }
    let variant = or_exit(cfg.mp2.linlccd_variant());
    let model = or_exit(cfg.local_model());
    let method = format!("LinLCCD({})", variant.as_str());
    let (aux_name, dfbs) = correlation_aux(cfg, mol);
    let frozen_core = cfg.mp2.frozen_core.resolve(mol);
    let (e_corr, local) = match &model {
        None => {
            let cc_config = CcConfig {
                frozen_core,
                memory_budget_bytes: budget_bytes,
                ..Default::default()
            };
            let cc =
                linlccd(mol, prep, &dfbs, op, result, &cc_config, variant).unwrap_or_else(|e| {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                });
            println!("{}", model_label(&method, None));
            (cc.correlation_energy, None)
        }
        Some(m) => {
            use ferric_cc::linlccd_amplitude::{amplitude_linlccd, AmplitudeLinLccdConfig};
            let eps = m.eps[0];
            let r = amplitude_linlccd(
                mol,
                prep,
                bs,
                &dfbs,
                op,
                result,
                &AmplitudeLinLccdConfig {
                    eps,
                    frozen_core,
                    eri3_budget_bytes: budget_bytes,
                    ..Default::default()
                },
                variant,
            )
            .unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            let local = LocalPrint {
                eps,
                keep_fraction: r.keep_fraction,
                integral_direct: false,
            };
            println!("{}", model_label(&method, Some(&local)));
            println!(
                "  keep {:.4}  cg {}  relres {:.2e}  converged {}",
                r.keep_fraction, r.cg_iterations, r.cg_relres, r.cg_converged
            );
            if !r.cg_converged {
                eprintln!(
                    "warning: local LinLCCD PCG did not converge (relres {:.2e} after {} \
                     iterations)",
                    r.cg_relres, r.cg_iterations
                );
            }
            (r.e_corr, Some(local))
        }
    };
    // `[local] reference = true`: the exact LinLCCD of the same variant (the
    // canonical solve), and the local error against it.
    let e_ref = match &model {
        Some(m) if m.reference => Some(
            linlccd(
                mol,
                prep,
                &dfbs,
                op,
                result,
                &CcConfig {
                    frozen_core,
                    memory_budget_bytes: budget_bytes,
                    ..Default::default()
                },
                variant,
            )
            .unwrap_or_else(|e| {
                eprintln!("error: exact LinLCCD reference: {e}");
                std::process::exit(1);
            })
            .correlation_energy,
        ),
        _ => None,
    };
    if model.is_some() {
        for line in local_reference_lines(
            e_corr,
            e_ref,
            "E_corr(exact)",
            "threshold error",
            "~linear in eps",
        ) {
            println!("{line}");
        }
    }
    println!(
        "{method}/{} (aux: {}) on {}",
        bs.name, aux_name, cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!("  RHF energy = {:.10} Hartree", result.energy);
    println!("  LinLCCD corr = {e_corr:.10} Hartree");
    println!("  Total      = {:.10} Hartree", result.energy + e_corr);
    if let Some(rl) = ferric_scf::runlog::log() {
        rl.result(
            "linlccd",
            result.energy + e_corr,
            serde_json::json!({
                "variant": variant.as_str(),
                "e_corr": e_corr,
                // null unless a local run opted in to the exact reference
                "e_corr_exact": e_ref,
                "e_scf_reference": result.energy,
                "scf_converged": result.converged,
                "local": local_json(local.as_ref()),
            }),
        );
    }
}

/// The `[mp2] auxbasis` (default [`config::DEFAULT_CORRELATION_AUX`]) name and
/// its prepared basis, exiting with the error on failure. Shared by the four
/// arms below, which resolve their RI aux exactly as `run_ccsd` does.
fn correlation_aux(cfg: &Config, mol: &Molecule) -> (String, PreparedBasis) {
    let aux_name = cfg
        .mp2
        .auxbasis
        .as_deref()
        .unwrap_or(config::DEFAULT_CORRELATION_AUX)
        .to_string();
    let aux_bs = basis::bundled(&aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    (aux_name, dfbs)
}

/// Refuse a non-restricted reference for a closed-shell-only arm with a clear
/// message instead of letting an `eps_r()`/`mos_r()` assert fire as a panic.
/// `Config::validate_multiplicity` already refuses multiplicity > 1 up front;
/// this is the backstop.
fn require_restricted(result: &ferric_scf::result::ScfResult, kind: &str) {
    if !matches!(result.spin, ferric_scf::result::Spin::Restricted) {
        eprintln!("error: method.kind = \"{kind}\" requires a closed-shell (RHF) reference");
        std::process::exit(1);
    }
}

/// `method.kind = "ccd"`. RI-CCD (`ferric_cc::ccd::ccd`, spin-orbital) on
/// the converged closed-shell reference; aux and frozen core from `[mp2]`
/// like `ccsd`. Same library call as Python `run_ccd`.
fn run_ccd(
    cfg: &Config,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    op: Operator,
    result: &ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
) {
    require_restricted(result, "ccd");
    let (aux_name, dfbs) = correlation_aux(cfg, mol);
    let cc_config = CcConfig {
        frozen_core: cfg.mp2.frozen_core.resolve(mol),
        memory_budget_bytes: budget_bytes,
        ..Default::default()
    };
    let cc_result = ccd(mol, prep, &dfbs, op, result, &cc_config).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let total = result.energy + cc_result.correlation_energy;
    println!(
        "CCD/{} (aux: {}) on {}",
        bs.name, aux_name, cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!("  RHF energy = {:.10} Hartree", result.energy);
    println!(
        "  CCD corr   = {:.10} Hartree",
        cc_result.correlation_energy
    );
    println!("  Total      = {total:.10} Hartree");
    if let Some(rl) = ferric_scf::runlog::log() {
        rl.result(
            "ccd",
            total,
            serde_json::json!({
                "e_corr": cc_result.correlation_energy,
                "e_scf_reference": result.energy,
                "scf_converged": result.converged,
            }),
        );
    }
}

/// `method.kind = "ccsd(t)"`. Spin-adapted closed-shell CCSD
/// (`ccsd_closed_shell`) whose SPATIAL amplitudes feed the spin-adapted (T)
/// (`ccsd_t_closed_shell`) directly -- the same pair of calls as Python
/// `run_ccsd_t`. Prints E_CCSD (correlation), E_(T) and the total
/// `E_RHF + E_CCSD + E_(T)`.
///
/// The TOML spelling is `kind = "ccsd(t)"`: parentheses are ordinary
/// characters inside a TOML string, and it is the name the method goes by.
fn run_ccsd_t(
    cfg: &Config,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    op: Operator,
    result: &ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
) {
    require_restricted(result, "ccsd(t)");
    let (aux_name, dfbs) = correlation_aux(cfg, mol);
    let cc_config = CcConfig {
        frozen_core: cfg.mp2.frozen_core.resolve(mol),
        memory_budget_bytes: budget_bytes,
        ..Default::default()
    };
    let cc_result =
        ccsd_closed_shell(mol, prep, &dfbs, op, result, &cc_config).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });
    let e_t = ccsd_t_closed_shell(mol, prep, &dfbs, op, result, &cc_result, &cc_config)
        .unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });
    let e_ccsd = cc_result.correlation_energy;
    let total = result.energy + e_ccsd + e_t;
    println!(
        "CCSD(T)/{} (aux: {}, spin-adapted) on {}",
        bs.name, aux_name, cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!("  RHF energy = {:.10} Hartree", result.energy);
    println!("  CCSD corr  = {e_ccsd:.10} Hartree");
    println!("  (T) corr   = {e_t:.10} Hartree");
    println!("  Total      = {total:.10} Hartree");
    if let Some(rl) = ferric_scf::runlog::log() {
        rl.result(
            "ccsd(t)",
            total,
            serde_json::json!({
                "e_corr": e_ccsd + e_t,
                "e_ccsd_corr": e_ccsd,
                "e_t": e_t,
                "e_scf_reference": result.energy,
                "scf_converged": result.converged,
            }),
        );
    }
}

/// `method.kind = "drpa"`: direct RPA correlation (dRPA@HF) by the drCCD
/// Riccati solve on localized orbitals (`ferric_mp2::drpa_amplitude`;
/// closed-shell).
///
/// EXACT by default: the Riccati solve with nothing truncated (ε = 0), which
/// the library anchors to the canonical plasmon formula (<= 1e-12). Riccati,
/// plasmon and full-rank PDEP (`pdep-rpa`) are algorithms for the same exact
/// dRPA energy. The exact path's memory grows as `no^3·nv^2` (the ring-product
/// plan), so [`preflight_exact_drpa`] refuses a run that cannot fit before
/// the SCF.
///
/// With `[local] scheme = "amplitude-threshold"`: the amplitude-threshold
/// dRPA at `eps` (one-sided, ~linear-in-ε error; not variational), `reference`
/// (opt-in canonical plasmon reference) and `eps_sweep` (several ε on ONE SCF
/// and ONE ε-independent localized assembly via `amplitude_drpa_scan_timed`,
/// the `r0_sweep` pattern).
///
/// The fixed-point accelerators match the Python binding's defaults (DIIS
/// subspace 8, ε-linked stopping tolerance factor 0.1), so a CLI run and
/// `run_drpa(...)` with default kwargs solve the same equations the same way.
/// The ε-link is a no-op at ε = 0.
fn run_drpa(
    cfg: &Config,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    op: Operator,
    result: &ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
) {
    use ferric_mp2::drpa_amplitude::{
        amplitude_drpa, amplitude_drpa_scan_timed, AmplitudeDrpaConfig,
    };
    require_restricted(result, "drpa");
    let model = or_exit(cfg.local_model());
    let (points, is_sweep, want_ref) = match &model {
        None => (vec![0.0], false, false),
        Some(m) => (m.eps.clone(), m.is_sweep, m.reference),
    };
    let (aux_name, dfbs) = correlation_aux(cfg, mol);
    let base = AmplitudeDrpaConfig {
        eps: points[0],
        frozen_core: cfg.mp2.frozen_core.resolve(mol),
        eri3_budget_bytes: budget_bytes,
        compute_reference: want_ref,
        diis: Some(DRPA_DIIS_SUBSPACE),
        eps_rtol_factor: Some(0.1),
        ..Default::default()
    };
    fn fail(e: ferric_core::FerricError, exact: bool) -> ! {
        eprintln!("error: {e}");
        if exact {
            eprintln!("{EXACT_DRPA_MEMORY_HINT}");
        }
        std::process::exit(1);
    }
    let exact = model.is_none();
    let results = if is_sweep {
        let (rs, prefix_wall_s, _) =
            amplitude_drpa_scan_timed(mol, prep, bs, &dfbs, op, result, &base, &points)
                .unwrap_or_else(|e| fail(e, exact));
        eprintln!(
            "[ferric] [local] eps_sweep: {} points on one SCF + one localized assembly \
             ({prefix_wall_s:.2} s shared)",
            points.len()
        );
        rs
    } else {
        vec![amplitude_drpa(mol, prep, bs, &dfbs, op, result, &base)
            .unwrap_or_else(|e| fail(e, exact))]
    };
    let n_points = points.len();
    for (k, (r, eps)) in results.iter().zip(&points).enumerate() {
        if n_points > 1 {
            println!(
                "\n===== drpa eps sweep point {}/{}: eps = {eps:.1e} =====",
                k + 1,
                n_points
            );
        }
        let local = model.as_ref().map(|m| LocalPrint {
            eps: *eps,
            keep_fraction: r.keep_fraction,
            integral_direct: m.direct.is_some(),
        });
        println!("{}", model_label("dRPA", local.as_ref()));
        println!("  basis / aux           = {} / {aux_name}", bs.name);
        println!("  RHF energy            = {:.10} Ha", result.energy);
        println!("  E_corr(dRPA)          = {:.10} Ha", r.e_corr);
        let e_ref = want_ref.then_some(r.e_corr_plasmon_canonical);
        if model.is_some() {
            for line in local_reference_lines(
                r.e_corr,
                e_ref,
                "E_corr(canonical)",
                "threshold error",
                "~linear in eps; not variational",
            ) {
                println!("{line}");
            }
        }
        println!("  total energy          = {:.10} Ha", r.e_total);
        println!(
            "  keep {:.4}  pairs {:.3}  iterations {}  relres {:.2e}  converged {}",
            r.keep_fraction, r.pair_fraction, r.iterations, r.relres, r.converged
        );
        if !r.converged {
            eprintln!(
                "warning: drpa fixed point did not converge at eps = {eps:.1e} \
                 (relres {:.2e} after {} iterations)",
                r.relres, r.iterations
            );
        }
        if let Some(rl) = ferric_scf::runlog::log() {
            rl.result(
                "drpa",
                r.e_total,
                serde_json::json!({
                    "eps": eps,
                    "e_corr": r.e_corr,
                    // null when the opt-in reference was not computed
                    "e_corr_plasmon_canonical": e_ref,
                    "converged": r.converged,
                    "e_scf_reference": result.energy,
                    "scf_converged": result.converged,
                    "local": local_json(local.as_ref()),
                }),
            );
        }
    }
}

/// DIIS subspace of the CLI's dRPA Riccati solve (the Python binding's
/// default). Shared by [`run_drpa`] and [`preflight_exact_drpa`], whose
/// memory estimate depends on it.
const DRPA_DIIS_SUBSPACE: usize = 8;

/// Where an exact dRPA that does not fit should go instead.
const EXACT_DRPA_MEMORY_HINT: &str =
    "hint: exact dRPA through the Riccati solve holds the full no^3*nv^2 ring-product plan. \
     method.kind = \"pdep-rpa\" with [rpa] trunc_thresh = 0 (full rank) computes the same \
     exact dRPA energy (to its frequency-quadrature error) at far lower memory; \
     [local] scheme = \"amplitude-threshold\" with a stated eps is the local approximation.";

/// Refuse an EXACT `drpa` run whose Riccati solve cannot fit the memory
/// budget, BEFORE the SCF: the ε = 0 path's peak
/// (`ferric_mp2::drpa_amplitude::exact_drpa_peak_bytes`) is `no` times the
/// size of B in the ring-product plan alone, which made C12 thrash and then
/// be OOM-killed. The solve itself also hard-charges the memory pool (the
/// backstop); this check only moves the refusal ahead of the SCF and points
/// at the alternatives.
fn preflight_exact_drpa(cfg: &Config, mol: &Molecule, prep: &PreparedBasis, budget: Option<usize>) {
    if cfg.method.kind != "drpa" || !matches!(cfg.local_model(), Ok(None)) {
        return;
    }
    let nocc = (mol.nelec() as usize) / 2;
    let no = nocc.saturating_sub(cfg.mp2.frozen_core.resolve(mol));
    let nv = prep.nbasis().saturating_sub(nocc);
    let need = ferric_mp2::drpa_amplitude::exact_drpa_peak_bytes(no, nv, Some(DRPA_DIIS_SUBSPACE));
    let have = ferric_core::memory::resolve_budget(budget);
    if need > have.bytes {
        let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
        eprintln!(
            "error: exact dRPA (kind = \"drpa\", no [local]) needs ~{:.2} GiB for the Riccati \
             solve at no = {no}, nv = {nv}, over the {:.2} GiB memory budget [source: {}]; \
             refused before the SCF.",
            gib(need),
            gib(have.bytes),
            have.source.label()
        );
        eprintln!("{EXACT_DRPA_MEMORY_HINT}");
        std::process::exit(1);
    }
}

/// `method.kind = "wb97x-l-v"`. The ωB97X-L-V double hybrid.
///
/// Unlike every other correlated arm, this one converges its own Kohn-Sham
/// reference: [`run_wb97x_l_v`] forces `xc = "wB97X-L-V"` and drives
/// `ksdft_ladder` itself, then computes the SR-LinLCCD(hh) correction on those
/// frozen orbitals. It hard-errors on an unconverged reference — that guard is
/// deliberate (an unconverged KS density yields a plausible-looking but
/// meaningless correlation energy), so it is surfaced as a fatal error here
/// rather than downgraded to a warning.
///
/// λ and ω default to the published values (0.6 and 0.1 Bohr⁻¹) carried by
/// `DoubleHybridConfig::default()`; `[dft] lambda` / `[dft] omega` override
/// them individually, so an omitted key always yields the published parameter.
#[allow(clippy::too_many_arguments)]
fn run_wb97x_l_v_arm(
    cfg: &Config,
    ctx: &ParallelContext,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    rhf_config: &RhfConfig,
    budget_bytes: Option<usize>,
) {
    let aux_name = cfg
        .mp2
        .auxbasis
        .as_deref()
        .unwrap_or(config::DEFAULT_CORRELATION_AUX);
    let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    // Start from the published defaults and OVERRIDE only what the user set, so
    // an omitted `[dft] lambda`/`omega` gives the paper's parameter rather than
    // a zero from a `..Default::default()`-less struct literal.
    let mut dh_cfg = DoubleHybridConfig {
        cc: CcConfig {
            frozen_core: cfg.mp2.frozen_core.resolve(mol),
            memory_budget_bytes: budget_bytes,
            ..DoubleHybridConfig::default().cc
        },
        ..Default::default()
    };
    if let Some(lambda) = cfg.dft.lambda {
        dh_cfg.lambda = lambda;
    }
    if let Some(omega) = cfg.dft.omega {
        dh_cfg.omega = omega;
    }
    if let Some(f) = cfg.dft.functional.as_deref() {
        // `run_wb97x_l_v` overwrites `xc` unconditionally. Say so rather than
        // letting a user believe `[dft] functional = "PBE"` did anything.
        if !f.eq_ignore_ascii_case("wB97X-L-V") {
            eprintln!(
                "warning: [dft] functional = \"{f}\" is ignored for method.kind = \"wb97x-l-v\"; \
                 the double hybrid always converges its own wB97X-L-V reference"
            );
        }
    }
    let (dh, ks) = run_wb97x_l_v(ctx, mol, prep, &dfbs, bounds, rhf_config, &dh_cfg)
        .unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });
    println!(
        "wB97X-L-V/{} (aux: {}, lambda={:.4}, omega={:.4} Bohr^-1) on {}",
        bs.name, aux_name, dh.lambda, dh.omega, cfg.molecule.xyz
    );
    println!("  nbasis       = {}", prep.nbasis());
    println!("  SCF iters    = {}", ks.iterations);
    // Components are printed separately on purpose: the DFT and WFT halves have
    // very different reliability characteristics, and one collapsed number makes
    // a bad SCF indistinguishable from a bad amplitude solve.
    println!("  E_KS         = {:.10} Hartree", dh.e_ks);
    println!("  E_c LinLCCD  = {:.10} Hartree", dh.e_c_wft);
    println!("  lambda*E_c   = {:.10} Hartree", dh.e_c_scaled);
    println!("  Total        = {:.10} Hartree", dh.total_energy);
}

/// `method.kind = "b2plyp"` or `"dsd-pbep86"`. MP2-based double hybrids.
///
/// Like wB97X-L-V, these converge their own KS reference (the SCF functional
/// is baked into the double-hybrid definition), then add scaled RI-MP2
/// correlation. Returns early — must not fall through to the generic SCF path.
#[allow(clippy::too_many_arguments)]
fn run_mp2_double_hybrid_arm(
    cfg: &Config,
    ctx: &ParallelContext,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    rhf_config: &RhfConfig,
    budget_bytes: Option<usize>,
    method: &str,
) {
    let dh_kind = match method {
        "b2plyp" => DoubleHybridKind::B2plyp,
        "dsd-pbep86" => DoubleHybridKind::DsdPbep86,
        _ => unreachable!(),
    };
    let aux_name = cfg
        .mp2
        .auxbasis
        .as_deref()
        .unwrap_or(config::DEFAULT_CORRELATION_AUX);
    let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });

    // Same courtesy as the wb97x-l-v arm: the double hybrid always converges
    // its own functional, so a different `[dft] functional` is overridden --
    // say so rather than let the user believe it did anything.
    if let Some(f) = cfg.dft.functional.as_deref() {
        if !f.eq_ignore_ascii_case(dh_kind.xc_name()) {
            eprintln!(
                "warning: [dft] functional = \"{f}\" is ignored for method.kind = \"{method}\"; \
                 the double hybrid always converges its own {} reference",
                dh_kind.xc_name()
            );
        }
    }
    // `rhf_config.df_*_aux` already carry `[scf] df_j_aux`/`df_k_aux` through
    // the shared spelling parser (`""`/"exact"/"none"/"off" -> the `""`
    // no-fit sentinel, a name -> that aux). Only an OMITTED key falls back to
    // RI-JK via def2-universal-jkfit -- the same rule `ksdft` and
    // `run_wb97x_l_v` apply. These arms used to overwrite both keys
    // unconditionally, so `[scf] df_j_aux` was silently ignored here.
    let mut ks_cfg = rhf_config.clone();
    ks_cfg.xc = Some(dh_kind.xc_name().to_string());
    ks_cfg.df_j_aux = config::jk_aux_or_default(&rhf_config.df_j_aux);
    ks_cfg.df_k_aux = config::jk_aux_or_default(&rhf_config.df_k_aux);
    log_jk_path(&ks_cfg, false);

    let ladder = ferric_scf::ladder::ksdft_ladder(&ks_cfg);
    let lr =
        ferric_scf::ladder::solve_rhf_ladder(ctx, mol, prep, Operator::coulomb(), bounds, &ladder)
            .unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
    let ks = lr.result;
    if !ks.converged {
        eprintln!(
            "error: {} SCF did not converge after {} iterations",
            method, ks.iterations
        );
        std::process::exit(1);
    }

    let mut mp2_cfg = dh_kind.mp2_config();
    mp2_cfg.frozen_core = cfg.mp2.frozen_core.resolve(mol);
    mp2_cfg.memory_budget_bytes = budget_bytes;

    let r = mp2_double_hybrid(mol, prep, &dfbs, &ks, &mp2_cfg).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    println!(
        "{}/{} (aux: {}) on {}",
        method.to_uppercase(),
        bs.name,
        aux_name,
        cfg.molecule.xyz
    );
    println!("  nbasis       = {}", prep.nbasis());
    println!("  SCF iters    = {}", ks.iterations);
    println!("  E_KS         = {:.10} Hartree", r.e_ks);
    println!("  E_OS (raw)   = {:.10} Hartree", r.spin_components.e_os);
    println!("  E_SS (raw)   = {:.10} Hartree", r.spin_components.e_ss);
    println!(
        "  scaled corr  = {:.10} Hartree (c_os={}, c_ss={})",
        r.e_corr_scaled, r.c_os, r.c_ss
    );
    println!("  Total        = {:.10} Hartree", r.total_energy);
}

/// `method.kind = "laplace-mp2"`. Extracted verbatim from the former
/// `main()` `"laplace-mp2" => { ... }` match arm.
fn run_laplace_mp2(
    cfg: &Config,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    op: Operator,
    result: &ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
) {
    let aux_name = cfg
        .mp2
        .auxbasis
        .as_deref()
        .unwrap_or(config::DEFAULT_CORRELATION_AUX);
    let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let n_quad = cfg.mp2.n_quad.unwrap_or(7);
    let lap_result = laplace_ri_mp2(
        mol,
        prep,
        &dfbs,
        op,
        result,
        n_quad,
        cfg.mp2.frozen_core.resolve(mol),
        budget_bytes,
    )
    .unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    println!(
        "Laplace RI-MP2/{} (aux: {}, n_quad={}) on {}",
        bs.name, aux_name, n_quad, cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!("  RHF energy = {:.10} Hartree", result.energy);
    println!("  MP2 corr   = {:.10} Hartree", lap_result.mp2_corr);
    println!("  E_OS       = {:.10} Hartree", lap_result.e_os);
    println!("  E_SS       = {:.10} Hartree", lap_result.e_ss);
    println!("  Total      = {:.10} Hartree", lap_result.total_energy);
    if let Some(rl) = ferric_scf::runlog::log() {
        rl.result(
            "laplace-mp2",
            lap_result.total_energy,
            serde_json::json!({
                "e_corr": lap_result.mp2_corr,
                "e_scf_reference": result.energy,
                "scf_converged": result.converged,
            }),
        );
    }
}

/// `method.kind = "laplace-sos-mp2"`.
///
/// Scaled-opposite-spin MP2 via the Laplace transform. `[mp2] c_os` selects the
/// scaling (default 1.3, Jung/Head-Gordon); `c_os = 1.0` recovers the bare
/// opposite-spin energy, which is the hard internal reference the tests use.
/// `[mp2] sos_formulation` picks the MO or AO algebra — same quantity either
/// way, so it is an implementation choice, not a physics one.
fn run_laplace_sos_mp2(
    cfg: &Config,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    op: Operator,
    result: &ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
) {
    let aux_name = cfg
        .mp2
        .auxbasis
        .as_deref()
        .unwrap_or(config::DEFAULT_CORRELATION_AUX);
    let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    // Strict parse: an unrecognized value errors rather than silently running
    // the default formulation.
    let formulation = SosFormulation::parse_config_str(
        cfg.mp2.sos_formulation.as_deref(),
        cfg.mp2.domain_cutoff_bohr,
    )
    .unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    // NOTE: `c_ss` is deliberately NOT read here. SOS-MP2 *is* the c_ss = 0
    // limit — that is what makes the Laplace denominator factorize — so a
    // `c_ss` in the TOML would be silently ignored. Warn instead of lying.
    if cfg.mp2.c_ss.is_some() {
        eprintln!(
            "warning: [mp2] c_ss is ignored for laplace-sos-mp2 — SOS-MP2 is the \
             c_ss = 0 limit by construction (that is what makes the Laplace form \
             factorize). Use method.kind = \"scs-mp2\" if you want a same-spin term."
        );
    }
    let sos_cfg = SosMp2Config {
        c_os: cfg.mp2.c_os.unwrap_or(1.3),
        frozen_core: cfg.mp2.frozen_core.resolve(mol),
        n_quad: cfg.mp2.n_quad.unwrap_or(7),
        memory_budget_bytes: budget_bytes,
        domain_cutoff_bohr: cfg.mp2.domain_cutoff_bohr,
    };
    let sos =
        laplace_sos_mp2(mol, prep, &dfbs, op, result, &sos_cfg, formulation).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });
    let formulation_label = match formulation {
        SosFormulation::Mo => "MO".to_string(),
        SosFormulation::Ao => "AO (pseudo-density)".to_string(),
        SosFormulation::AoSparse(r) => {
            format!("AO sparse, domain cutoff {r} Bohr — APPROXIMATE")
        }
    };
    println!(
        "Laplace SOS-MP2/{} (aux: {}, n_quad={}, {}) on {}",
        bs.name, aux_name, sos.n_quad, formulation_label, cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!("  RHF energy = {:.10} Hartree", result.energy);
    if let SosFormulation::AoSparse(r) = formulation {
        println!(
            "  NOTE: domain-restricted AO path (cutoff {r} Bohr) — this is an \
             APPROXIMATION to the exact AO/MO result, converging to it as the \
             cutoff grows. Cross-check against sos_formulation = \"ao\"."
        );
    }
    println!("  E_OS       = {:.10} Hartree  (unscaled)", sos.e_os);
    println!("  c_os       = {:.4}", sos.c_os);
    println!("  SOS corr   = {:.10} Hartree", sos.sos_corr);
    println!("  Total      = {:.10} Hartree", sos.total_energy);
    if let Some(rl) = ferric_scf::runlog::log() {
        rl.result(
            "laplace-sos-mp2",
            sos.total_energy,
            serde_json::json!({
                "e_corr": sos.sos_corr,
                "e_scf_reference": result.energy,
                "scf_converged": result.converged,
            }),
        );
    }
}

/// `method.kind = "pdep-rpa"`. Extracted verbatim from the former `main()`
/// `"pdep-rpa" => { ... }` match arm (body unchanged; only the surrounding
/// `&x` -> `x` reference-vs-value adjustments needed for the new parameter
/// list, and `Some(&proatom)` -> `Some(proatom)` since `proatom` is now
/// itself the `&dyn Fn` reference).
#[allow(clippy::too_many_arguments)]
fn run_pdep_rpa_arm(
    cfg: &Config,
    ctx: &ParallelContext,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf_config: &RhfConfig,
    result: ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
    proatom: &dyn Fn(i32, i32) -> Option<ferric_rpa::properties::RadialProatom>,
) {
    let aux_name = cfg
        .rpa
        .auxbasis
        .as_deref()
        .unwrap_or(config::DEFAULT_CORRELATION_AUX);
    let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let scheme = cfg.rpa.parse_quadrature().unwrap_or_else(|e| {
        eprintln!("config error: {e}");
        std::process::exit(1);
    });
    let rpa_cfg = PdepRpaConfig {
        frozen_core: cfg.rpa.frozen_core.resolve(mol),
        trunc_thresh: cfg.rpa.trunc_thresh.unwrap_or(1e-4),
        eigensolver_max_vecs: 0,
        eigensolver_conv_thresh: cfg.rpa.eigensolver_conv_thresh.unwrap_or(1e-6),
        quadrature: QuadratureConfig {
            scheme,
            n_points: cfg.rpa.n_quad.unwrap_or(20),
            u0: cfg.rpa.u0.unwrap_or(0.5),
        },
        sternheimer: SternheimerConfig::default(),
        run_diagnostics: cfg.rpa.run_diagnostics,
        eigensolver: ferric_rpa::Eigensolver::default(),
        chi0_backend: ferric_rpa::config::Chi0Backend::default(),
        chi0_sparsity: cfg.rpa.parse_chi0_sparsity().unwrap_or_else(|e| {
            eprintln!("config error: {e}");
            std::process::exit(1);
        }),
        memory_budget_bytes: budget_bytes,
        // CLI RPA energy + NPZ property export; the property paths that
        // consume the inverse-dielectric stack rebuild their own
        // dielectric, so energy-only here is correct (M9 gate).
        need_inv_dielectric_freq: false,
        // Verified: this arm reads only `e_rpa`, `e_rpa_dft_diag`, and
        // `eigenpotentials` off the RPA result — never `eigenvalues_freq`.
        // The NPZ export calls properties::pdep_polarizability_*, which run
        // their own PDEP-RPA with their own configs and so are unaffected.
        // Opting out skips the per-frequency diagonalization and takes the
        // LU log-det path for the correlation energy.
        need_eigenvalues_freq: false,
        verbose: cfg.scf.verbose,
    };
    // For open-shell molecules (multiplicity > 1) re-run with UHF + MOM so
    // the reference is converged, then dispatch to the unrestricted RPA.
    // Shadow `result` so the rest of the arm (NPZ export, properties) uses
    // the correct SCF density.
    let (rpa_result, ref_label, result) = if mol.multiplicity > 1 {
        let mut uhf_cfg = rhf_config.clone();
        // MOM after 5 DIIS iters prevents orbital reordering on open-shell atoms.
        uhf_cfg.mom_after_iter = 5;
        let uhf_result = solve_uhf(ctx, mol, prep, bounds, &uhf_cfg).unwrap_or_else(|e| {
            eprintln!("error (UHF): {e}");
            std::process::exit(1);
        });
        let rr = ferric_rpa::run_u_pdep_rpa(mol, prep, &dfbs, op, &uhf_result, &rpa_cfg)
            .unwrap_or_else(|e| {
                eprintln!("error (U-PDEP-RPA): {e}");
                std::process::exit(1);
            });
        (rr, "UHF", uhf_result)
    } else {
        let rr = run_pdep_rpa(mol, prep, &dfbs, op, &result, &rpa_cfg).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });
        (rr, "RHF", result)
    };
    if !rpa_result.eigensolver_converged {
        eprintln!(
            "warning: PDEP-RPA eigensolver did not fully converge (best-effort Ritz pairs; \
                 eigenvalues_static/eigenpotentials below are not verified to residual tolerance)"
        );
    }
    println!(
        "PDEP-RPA/{} (aux: {}) on {}",
        bs.name, aux_name, cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!(
        "{ref_label} energy:            {:>20.10} Hartree",
        result.energy
    );
    println!("RPA correlation:       {:>20.10} Hartree", rpa_result.e_rpa);
    println!(
        "Total ({ref_label}+RPA):       {:>20.10} Hartree",
        result.energy + rpa_result.e_rpa
    );
    println!(
        "Eigenpotentials kept:  {} / {}",
        rpa_result.n_eigenpotentials,
        rpa_result.eigenvalues_static.len()
    );
    if let Some(e_diag) = rpa_result.e_rpa_dft_diag {
        println!("RI-dRPA check:         {:>20.10} Hartree", e_diag);
    }
    if let Some(prefix) = cfg.rpa.export_eigpot_prefix.as_deref() {
        use ferric_export::cube::GridSpec;
        use ferric_export::export_basis_function_cube;
        let spacing = cfg.rpa.cube_spacing.unwrap_or(0.2);
        let margin = cfg.rpa.cube_margin.unwrap_or(4.0);
        let n_export = cfg
            .rpa
            .export_eigpot_count
            .unwrap_or(10)
            .min(rpa_result.n_eigenpotentials);
        let grid = GridSpec::bounding_box(mol, margin, spacing);
        println!(
            "Exporting {} eigenpotential cubes (grid {}×{}×{}, spacing {} Bohr)…",
            n_export, grid.n_x, grid.n_y, grid.n_z, spacing
        );
        for alpha in 0..n_export {
            let coeffs: Vec<f64> = rpa_result
                .eigenpotentials
                .column(alpha)
                .iter()
                .copied()
                .collect();
            let lam = rpa_result.eigenvalues_static[alpha];
            let path = format!("{prefix}_eigpot_{:03}.cube", alpha);
            let comment = format!("PDEP eigenpotential α={alpha} λ(0)={lam:.6} (basis {aux_name})");
            if let Err(e) =
                export_basis_function_cube(&path, mol, &aux_bs, &grid, &coeffs, &comment)
            {
                eprintln!("  warning: failed to write {}: {}", path, e);
            } else {
                println!("  wrote {} (λ(0)={:.6})", path, lam);
            }
        }
    }
    // NPZ feature bundle for diffusion-model export.
    if let Some(npz_path) = cfg.rpa.export_npz.as_deref() {
        use ferric_export::export_npz;
        use ferric_export::ml::{
            C6Export, C6Provenance, ChargeSchemes, DispersionBundle, NpzBundle,
            PolarizabilityBundle,
        };
        use ferric_rpa::properties::{
            chelpg_and_resp_charges, chelpg_charges, electric_field_at_atoms, esp_at_atoms,
            hirshfeld_charges, lowdin_charges, mulliken_charges, pdep_polarizability_becke,
            pdep_polarizability_static, resp_charges,
        };
        use ndarray::Array2;

        let compute_esp = cfg.rpa.compute_esp.unwrap_or(true);
        let compute_pol = cfg.rpa.compute_polarizability.unwrap_or(true);
        let compute_ef = cfg.rpa.compute_electric_field.unwrap_or(true);
        let compute_alpha_atomic = cfg.rpa.compute_alpha_atomic.unwrap_or(true);

        // Properties that were REQUESTED and did not make it into the
        // bundle. Every arm below is `Err(e) => { warn; None }`, and the
        // bundle is written regardless with the gaps as absent arrays --
        // which used to leave the process exiting 0 on an incomplete file.
        // See `RpaCfg::allow_partial_npz` for the incident.
        //
        // Recording the REQUESTED-and-failed set, rather than counting
        // absent fields in the finished bundle, is the distinction that
        // keeps a deliberately disabled property (`compute_c6 = false`)
        // from reading as a failure.
        let mut npz_gaps: Vec<String> = Vec::new();

        let coords_arr = {
            let mut a = Array2::<f64>::zeros((mol.atoms.len(), 3));
            for (i, atom) in mol.atoms.iter().enumerate() {
                a[(i, 0)] = atom.x;
                a[(i, 1)] = atom.y;
                a[(i, 2)] = atom.zpos;
            }
            a
        };
        let znums: Vec<usize> = mol.atoms.iter().map(|a| a.z as usize).collect();

        let esp_vec = if compute_esp {
            match esp_at_atoms(mol, prep, result.density_total()) {
                Ok(v) => Some(v),
                Err(e) => {
                    eprintln!("warning: esp_at_atoms failed: {e}");
                    npz_gaps.push(format!("esp_atoms: {e}"));
                    None
                }
            }
        } else {
            None
        };

        // `esp_points` is exported as an (npts, 3) Array2, so materialize
        // the coordinate list into one here rather than at the call site.
        let mut esp_surface_pts: Option<ndarray::Array2<f64>> = None;
        let esp_surface = if cfg.rpa.compute_esp_surface.unwrap_or(false) {
            let scale = cfg.rpa.esp_surface_vdw_scale.unwrap_or(1.4);
            let nang = cfg.rpa.esp_surface_n_angular.unwrap_or(110);
            match ferric_scf::properties::esp_on_surface(
                mol,
                prep,
                result.density_total(),
                scale,
                nang,
            ) {
                Ok((pts, v)) => {
                    let mut arr = ndarray::Array2::<f64>::zeros((pts.len(), 3));
                    for (i, p) in pts.iter().enumerate() {
                        arr[(i, 0)] = p[0];
                        arr[(i, 1)] = p[1];
                        arr[(i, 2)] = p[2];
                    }
                    esp_surface_pts = Some(arr);
                    Some((pts, v))
                }
                Err(e) => {
                    eprintln!("warning: esp_on_surface failed: {e}");
                    npz_gaps.push(format!("esp_surface: {e}"));
                    None
                }
            }
        } else {
            None
        };

        let ef_vec = if compute_ef {
            match electric_field_at_atoms(mol, prep, result.density_total()) {
                Ok(v) => Some(v),
                Err(e) => {
                    eprintln!("warning: electric_field_at_atoms failed: {e}");
                    npz_gaps.push(format!("electric_field: {e}"));
                    None
                }
            }
        } else {
            None
        };

        let alpha_arr = if compute_pol {
            match pdep_polarizability_static(mol, prep, &dfbs, &result, op, &rpa_cfg) {
                Ok(p) => {
                    println!(
                        "Polarizability α (a.u.):  iso={:.4}, principal=[{:.4}, {:.4}, {:.4}]",
                        p.iso, p.principal[0], p.principal[1], p.principal[2]
                    );
                    Some(p.tensor)
                }
                Err(e) => {
                    eprintln!("warning: polarizability failed: {e}");
                    npz_gaps.push(format!("alpha_tensor (polarizability): {e}"));
                    None
                }
            }
        } else {
            None
        };

        let alpha_atomic_vec = if compute_alpha_atomic {
            match pdep_polarizability_becke(mol, prep, bs, &dfbs, &result, op, &rpa_cfg) {
                Ok(v) => {
                    println!(
                        "Per-atom intrinsic Becke α (iso, a.u.): {:?}",
                        v.iter()
                            .map(|t| (t[0][0] + t[1][1] + t[2][2]) / 3.0)
                            .collect::<Vec<_>>()
                    );
                    Some(v)
                }
                Err(e) => {
                    eprintln!("warning: per-atom α (Becke) failed: {e}");
                    npz_gaps.push(format!("alpha_atomic (per-atom α): {e}"));
                    None
                }
            }
        } else {
            None
        };

        // Charge-transfer remainder α_CT = α_mol − Σ_A α^A: emitted exactly
        // when both inputs were computed. Both come from the same response
        // (pdep_polarizability_static / pdep_polarizability_becke: same RI
        // kernel, same orbitals, frozen_core = 0), so the remainder is the
        // Krishtal charge-delocalization polarizability, not a definition gap.
        let alpha_ct_arr: Option<[[f64; 3]; 3]> =
            match (alpha_arr.as_ref(), alpha_atomic_vec.as_deref()) {
                (Some(mol_a), Some(per_atom)) => {
                    let ct = ferric_rpa::properties::charge_transfer_remainder(mol_a, per_atom);
                    println!(
                        "Charge-transfer α_CT (iso, a.u.): {:.4}",
                        (ct[0][0] + ct[1][1] + ct[2][2]) / 3.0
                    );
                    Some(ct)
                }
                _ => None,
            };

        let compute_dm = cfg.rpa.compute_density_matrix.unwrap_or(true);
        let dm_ref = if compute_dm {
            Some(result.density_total())
        } else {
            None
        };

        // Molecular dipole μ = −Tr(P·D) + Σ_A Z_A R_A of the total density
        // (QC ground truth vs partition-derived Löwdin/Hirshfeld dipoles).
        // Origin [0,0,0]; neutral molecules → origin-independent. Mirrors
        // ferric-mp2 ff_polar::mp2_dipole; P·D summed elementwise = Tr(P·D)
        // since both AO matrices are symmetric.
        let compute_dip = cfg.rpa.compute_dipole.unwrap_or(true);
        let dip_arr: Option<[f64; 3]> = if compute_dip {
            match ferric_scf::properties::dipole_moment(mol, prep, result.density_total()) {
                Ok(mu) => {
                    let mag = ferric_scf::properties::dipole_magnitude(&mu);
                    println!(
                        "dipole (e·a0): [{:.4}, {:.4}, {:.4}] |μ| = {:.4} ({:.4} D)",
                        mu[0],
                        mu[1],
                        mu[2],
                        mag,
                        mag * ferric_scf::properties::DEBYE_PER_AU
                    );
                    Some(mu)
                }
                Err(e) => {
                    eprintln!("warning: dipole failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        let compute_lq = cfg.rpa.compute_lowdin_charges.unwrap_or(true);
        let lq_vec = if compute_lq {
            match lowdin_charges(mol, prep, result.density_total()) {
                Ok(q) => {
                    println!(
                        "Löwdin charges (e): {:?}",
                        q.iter()
                            .map(|v| (v * 1e4).round() / 1e4)
                            .collect::<Vec<_>>()
                    );
                    Some(q)
                }
                Err(e) => {
                    eprintln!("warning: Löwdin charges failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        let compute_hq = cfg.rpa.compute_hirshfeld_charges.unwrap_or(true);
        let hq_vec = if compute_hq {
            match hirshfeld_charges(mol, bs, result.density_total(), Some(proatom)) {
                Ok(q) => {
                    println!(
                        "Hirshfeld charges (e): {:?}",
                        q.iter()
                            .map(|v| (v * 1e4).round() / 1e4)
                            .collect::<Vec<_>>()
                    );
                    Some(q)
                }
                Err(e) => {
                    eprintln!("warning: Hirshfeld charges failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        let compute_mq = cfg.rpa.compute_mulliken_charges.unwrap_or(true);
        let mq_vec = if compute_mq {
            match mulliken_charges(mol, prep, result.density_total()) {
                Ok(q) => {
                    println!(
                        "Mulliken charges (e): {:?}",
                        q.iter()
                            .map(|v| (v * 1e4).round() / 1e4)
                            .collect::<Vec<_>>()
                    );
                    Some(q)
                }
                Err(e) => {
                    eprintln!("warning: Mulliken charges failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        // CHELPG and RESP differ ONLY in the least-squares solve; both
        // evaluate the same molecular ESP over the same grid. When both are
        // requested (the default) share one grid — evaluating it twice cost
        // ~2.2 s per duplicate at benzene/def2-SVP on 12 threads.
        let compute_cq = cfg.rpa.compute_chelpg_charges.unwrap_or(true);
        let compute_rq = cfg.rpa.compute_resp_charges.unwrap_or(true);
        fn fmt_q(q: &[f64]) -> Vec<f64> {
            q.iter().map(|v| (v * 1e4).round() / 1e4).collect()
        }
        let (cq_vec, rq_vec) = match (compute_cq, compute_rq) {
            (true, true) => match chelpg_and_resp_charges(mol, prep, result.density_total()) {
                Ok((cq, rq)) => {
                    println!("CHELPG charges (e): {:?}", fmt_q(&cq));
                    println!("RESP charges (e): {:?}", fmt_q(&rq));
                    (Some(cq), Some(rq))
                }
                Err(e) => {
                    eprintln!("warning: CHELPG/RESP charges failed: {e}");
                    (None, None)
                }
            },
            (true, false) => match chelpg_charges(mol, prep, result.density_total()) {
                Ok(q) => {
                    println!("CHELPG charges (e): {:?}", fmt_q(&q));
                    (Some(q), None)
                }
                Err(e) => {
                    eprintln!("warning: CHELPG charges failed: {e}");
                    (None, None)
                }
            },
            (false, true) => match resp_charges(mol, prep, result.density_total()) {
                Ok(q) => {
                    println!("RESP charges (e): {:?}", fmt_q(&q));
                    (None, Some(q))
                }
                Err(e) => {
                    eprintln!("warning: RESP charges failed: {e}");
                    (None, None)
                }
            },
            (false, false) => (None, None),
        };

        // --- C6 dispersion (Phase 1: Tkatchenko-Scheffler model) ---
        let compute_c6 = cfg.rpa.compute_c6.unwrap_or(true);
        let mut c6_freqs_v: Vec<f64> = Vec::new();
        let mut c6_weights_v: Vec<f64> = Vec::new();
        let mut alpha_dyn_v: Vec<Vec<[[f64; 3]; 3]>> = Vec::new();
        // Dynamic charge-transfer remainder; PDEP source only (TS/MBD model
        // α(iω) has no charge-transfer term to remove).
        let mut alpha_ct_dyn_v: Option<Vec<[[f64; 3]; 3]>> = None;
        let mut c6_iso_opt: Option<ndarray::Array2<f64>> = None;
        let mut c6_aniso_v: Vec<Vec<[[f64; 3]; 3]>> = Vec::new();
        // Provenance for the per-atom C6 arrays, carried to the NPZ so an
        // untagged per-atom number never leaves ferric (a per-atom C6 is a
        // partition CONVENTION, not an observable — Becke vs Hirshfeld differ
        // by up to ~10x). Populated from the SAME `partition`/`c6_source`
        // values the computation actually ran with, below — never defaulted.
        let mut c6_partition_s: Option<&'static str> = None;
        let mut c6_source_s: Option<&'static str> = None;
        let mut c6_molecular_iso_v: f64 = 0.0;
        if compute_c6 {
            use ferric_rpa::dispersion::{
                casimir_polder_c6, pdep_dynamic_polarizability, ts_dynamic_polarizability,
                C6Source, DispersionPartition,
            };
            use ferric_rpa::properties::{
                atomic_effective_volumes_hirshfeld, pdep_polarizability_hirshfeld,
            };
            use ferric_rpa::quadrature::build_quadrature;

            // Strict parse: an unknown c6_source/c6_partition used to fall
            // through to TS/Becke silently, producing different numbers than
            // the user asked for.
            let c6_source = C6Source::parse_config_str(cfg.rpa.c6_source.as_deref())
                .unwrap_or_else(|e| {
                    eprintln!("config error: [rpa] {e}");
                    std::process::exit(1);
                });
            let partition = DispersionPartition::parse_config_str(cfg.rpa.c6_partition.as_deref())
                .unwrap_or_else(|e| {
                    eprintln!("config error: [rpa] {e}");
                    std::process::exit(1);
                })
                .unwrap_or_else(|| c6_source.default_partition());
            let use_pdep = c6_source == C6Source::Pdep;

            let res_opt = if use_pdep {
                // Phase 2: PDEP-RPA dynamic α(iω). Origin-independent for
                // the molecular total AND the per-atom intrinsic α^A
                // (atom-centred (r−R_A); bond-axis anisotropy is a
                // coupled/molecular property, not per-atom). Uses the
                // shared ad-hoc same-basis Hirshfeld proatom (built once
                // above) so the per-atom partition is basis-consistent.
                match pdep_dynamic_polarizability(
                    mol,
                    prep,
                    bs,
                    &dfbs,
                    &result,
                    op,
                    &rpa_cfg,
                    partition,
                    Some(proatom),
                ) {
                    Ok(dp) => {
                        let res = casimir_polder_c6(&dp);
                        println!(
                            "Computed PDEP-RPA C6: {} atoms, {} freqs; molecular C6 = {:.3} a.u.",
                            mol.atoms.len(),
                            dp.freqs.len(),
                            res.c6_molecular_iso
                        );
                        Some(res)
                    }
                    Err(e) => {
                        eprintln!("warning: PDEP-RPA C6 failed: {e}");
                        None
                    }
                }
            } else {
                // Phase 1: Tkatchenko-Scheffler single-pole model.
                // Any failure below warns and SKIPS C6 (None) — the old
                // fallbacks (zero α, unit volumes, unit ratios) exported
                // wrong numbers that looked like results.
                (|| -> Option<ferric_rpa::dispersion::C6Result> {
                    let alpha_res = if partition == DispersionPartition::Hirshfeld {
                        pdep_polarizability_hirshfeld(
                            mol,
                            prep,
                            bs,
                            &dfbs,
                            &result,
                            op,
                            &rpa_cfg,
                            Some(proatom),
                        )
                    } else {
                        match alpha_atomic_vec.as_ref() {
                            Some(v) => Ok(v.clone()),
                            None => pdep_polarizability_becke(
                                mol, prep, bs, &dfbs, &result, op, &rpa_cfg,
                            ),
                        }
                    };
                    let alpha_static: Vec<[[f64; 3]; 3]> = match alpha_res {
                        Ok(v) => v,
                        Err(e) => {
                            eprintln!("warning: TS C6 skipped — per-atom static α failed: {e}");
                            return None;
                        }
                    };
                    // TS volumes must always use Hirshfeld partition — TS was
                    // parameterized with Hirshfeld volumes (TS PRL 2009). Becke
                    // volumes blow up for π-system H atoms (vol_ratio >> 1)
                    // because Becke is atom-size-blind; Hirshfeld proatom weights
                    // correctly compress H relative to C. The c6_partition setting
                    // only governs the alpha_static shape tensor, not these volumes.
                    let vols = match atomic_effective_volumes_hirshfeld(
                        mol,
                        bs,
                        result.density_total(),
                        Some(proatom),
                    ) {
                        Ok(v) => v,
                        Err(e) => {
                            eprintln!(
                                "warning: TS C6 skipped — Hirshfeld effective volumes failed: {e}"
                            );
                            return None;
                        }
                    };
                    let z: Vec<usize> = mol.atoms.iter().map(|a| a.z as usize).collect();

                    // Free-atom vol_free from a live free-atom SCF in the same
                    // basis and SCF settings, then Hirshfeld on the isolated
                    // atom (weight 1 wherever the Slater ρ⁰ is above the 1e-12
                    // floor) on the SAME atom-centred Becke–Lebedev quadrature
                    // `atomic_effective_volumes_hirshfeld` just integrated the
                    // molecular volumes on, so the ratio is scale-consistent. Shared with MBD@rsSCS dispersion
                    // (`ferric_rpa::dispersion::live_free_atom_volume`, which
                    // documents the solve and its HF/UHF retry). A failure
                    // leaves no entry for that Z, and the loop below skips TS
                    // C6 with a warning.
                    let mut vol_free_computed: std::collections::HashMap<usize, f64> =
                        std::collections::HashMap::new();
                    for &zi in z.iter().collect::<std::collections::HashSet<_>>() {
                        if let Ok(vf) = ferric_rpa::dispersion::live_free_atom_volume(
                            ctx, zi, bs, op, rhf_config,
                        ) {
                            vol_free_computed.insert(zi, vf);
                        }
                    }

                    // vol_free comes ONLY from the live free-atom SCF above,
                    // computed on the SAME integration scale (same xc, same
                    // Hirshfeld quadrature) as the molecular vols[i] — the only
                    // number for which the ratio vols[i]/vf is physically
                    // meaningful. There is deliberately NO table fallback: the
                    // hardcoded ts_free_atom vol_free values were on a mismatched
                    // integration scale and (for Z outside {H,He,C,N,O,F,Ne})
                    // were never sourced — feeding one to this ratio silently
                    // degraded the C6 to a wrong number that looked like a result
                    // (verified 2026-07-17, docs/vol-free-verification.md; Si's
                    // table 60.0 was 42% low vs the live-SCF value, inflating
                    // every Si-containing molecule's TS C6). Per this repo's
                    // established TS/MBD honesty convention (2026-07-09:
                    // ts_atom_params / ts_dynamic_polarizability / mbd_screen all
                    // hard-error rather than fabricate a Z>18 value), a genuine
                    // live-SCF failure now SKIPS TS C6 with a clear warning —
                    // matching the Z>18 "no honest value to return" behavior —
                    // instead of substituting a scale-mismatched fallback.
                    let mut ratio = Vec::with_capacity(z.len());
                    for (i, &zi) in z.iter().enumerate() {
                        let sym = ferric_core::elements::z_to_symbol(zi as i32).unwrap_or("?");
                        let vf = match vol_free_computed.get(&zi).copied() {
                            Some(v) => v,
                            None => {
                                eprintln!(
                                    "warning: TS C6 skipped — live free-atom SCF failed for \
                                     {sym} (Z={zi}) and no scale-consistent free-atom volume \
                                     is available. The TS free-atom vol_free denominator MUST \
                                     come from a live SCF on the same integration scale as the \
                                     molecular volume; the old hardcoded-table fallback was \
                                     removed because it is on a mismatched scale and was never \
                                     sourced for most elements (project wiki: vol-free-verification.md). \
                                     Refusing to fabricate a C6 from a mismatched denominator \
                                     (same convention as the Z>18 hard-error path — see \
                                     ts_atom_params). Use \
                                     c6_source=\"pdep\" for a table-free dispersion source."
                                );
                                return None;
                            }
                        };
                        if vf <= 1e-10 {
                            eprintln!(
                                "warning: TS C6 skipped — degenerate free-atom volume \
                                 {vf:.3e} for {sym} (Z={zi})"
                            );
                            return None;
                        }
                        ratio.push(vols[i] / vf);
                    }
                    let (freqs, weights) = build_quadrature(&rpa_cfg.quadrature);
                    let is_mbd = c6_source == C6Source::Mbd;
                    let dp_res = if is_mbd {
                        let positions: Vec<[f64; 3]> =
                            mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();
                        ferric_rpa::dispersion::mbd_dynamic_polarizability(
                            &positions,
                            &z,
                            &ratio,
                            &alpha_static,
                            &freqs,
                            &weights,
                        )
                    } else {
                        ts_dynamic_polarizability(&z, &ratio, &alpha_static, &freqs, &weights)
                    };
                    let dp = match dp_res {
                        Ok(dp) => dp,
                        Err(e) => {
                            eprintln!(
                                "warning: {} C6 skipped: {e}",
                                if is_mbd { "MBD" } else { "TS" }
                            );
                            return None;
                        }
                    };
                    let ts_res = casimir_polder_c6(&dp);
                    println!(
                        "Computed {} C6: {} atoms; molecular C6 = {:.3} a.u.",
                        if is_mbd { "MBD" } else { "TS" },
                        z.len(),
                        ts_res.c6_molecular_iso
                    );
                    Some(ts_res)
                })()
            };

            if let Some(res) = res_opt {
                if use_pdep {
                    // molecular_dynamic_polarizability and the per-atom
                    // intrinsic α^A(iω) share the RI kernel, orbitals and
                    // frozen_core = 0, at the same frequencies.
                    match ferric_rpa::properties::charge_transfer_remainder_dynamic(
                        &res.per_atom_dynamic.molecular,
                        &res.per_atom_dynamic.per_atom,
                    ) {
                        Ok(ct) => alpha_ct_dyn_v = Some(ct),
                        Err(e) => {
                            eprintln!("warning: dynamic charge-transfer α_CT(iω) failed: {e}");
                            npz_gaps.push(format!("alpha_ct_dynamic: {e}"));
                        }
                    }
                }
                c6_freqs_v = res.per_atom_dynamic.freqs.clone();
                c6_weights_v = res.per_atom_dynamic.weights.clone();
                alpha_dyn_v = res.per_atom_dynamic.per_atom.clone();
                c6_iso_opt = Some(res.c6_iso_pair.clone());
                c6_aniso_v = res.c6_aniso_pair.clone();
                // Tag with the partition/source this run ACTUALLY used (the
                // strictly-parsed locals above, after the source-dependent
                // default was applied), not a literal.
                c6_partition_s = Some(partition.as_config_str());
                c6_source_s = Some(c6_source.as_config_str());
                c6_molecular_iso_v = res.c6_molecular_iso;
            } else {
                // Recorded HERE rather than in the arms above because the
                // TS branch computes inside a closure (which cannot also
                // borrow `npz_gaps` mutably). Reaching this point with
                // `compute_c6` true means C6 was requested and every path
                // to it warned and bailed; the specific reason is already
                // on stderr from those arms.
                npz_gaps.push(
                    "c6_iso/c6_aniso/alpha_atomic_dynamic (see the C6 warning above)".to_string(),
                );
            }
        }

        // second moments of orbitals + density: one-electron cost,
        // computed whenever the pieces are already in hand
        let orbital_moments_opt = if result.spin == ferric_scf::result::Spin::Restricted {
            ferric_integrals::oneelectron::orbital_moments(prep, result.mos_r()).ok()
        } else {
            None
        };
        let density_m2_opt = dm_ref.and_then(|d| {
            ferric_integrals::oneelectron::density_second_moment(prep, d, [0.0; 3])
                .ok()
                .map(|m| ndarray::Array2::from_shape_fn((3, 3), |(p, q)| m[p][q]))
        });

        let npz_bundle = NpzBundle {
            mo_coeffs: if result.spin == ferric_scf::result::Spin::Restricted {
                Some(result.mos_r())
            } else {
                None
            },
            orbital_energies: if result.spin == ferric_scf::result::Spin::Restricted {
                Some(result.eps_r())
            } else {
                None
            },
            pdep_eigenvectors: Some(&rpa_result.eigenpotentials),
            boys_coeffs: None,
            orbital_centers: orbital_moments_opt.as_ref().map(|(c, _)| c),
            orbital_spreads: orbital_moments_opt.as_ref().map(|(_, s)| s.as_slice()),
            density_second_moment: density_m2_opt.as_ref(),
            coords: Some(&coords_arr),
            atomic_numbers: Some(&znums),
            density_matrix: dm_ref,
            dipole: dip_arr.as_ref(),
            charges: ChargeSchemes {
                hirshfeld: hq_vec.as_deref(),
                lowdin: lq_vec.as_deref(),
                mulliken: mq_vec.as_deref(),
                chelpg: cq_vec.as_deref(),
                resp: rq_vec.as_deref(),
            },
            polarizability: PolarizabilityBundle {
                esp_atoms: esp_vec.as_deref(),
                // Surface ESP: enabled by `[rpa] compute_esp_surface`.
                // The shell is generated internally (Lebedev spheres at
                // vdW radii, buried points dropped), so the caller supplies
                // a scale and an order rather than a point set.
                esp_surface: esp_surface.as_ref().map(|(_, v)| v.as_slice()),
                esp_points: esp_surface_pts.as_ref(),
                alpha_tensor: alpha_arr.as_ref(),
                electric_field: ef_vec.as_deref(),
                alpha_atomic: alpha_atomic_vec.as_deref(),
                alpha_ct: alpha_ct_arr.as_ref(),
            },
            dispersion: DispersionBundle {
                // All-or-nothing, and the provenance is non-Option inside
                // `C6Export` — so this arm either supplies the per-atom
                // arrays WITH their partition/source, or writes no C6 at all.
                // `zip` here is the enforcement: provenance is only ever
                // `Some` on the same path that populated the arrays.
                c6: match (c6_iso_opt.as_ref(), c6_partition_s.zip(c6_source_s)) {
                    (Some(iso), Some((partition, source))) => Some(C6Export {
                        provenance: C6Provenance { partition, source },
                        c6_freqs: c6_freqs_v.as_slice(),
                        c6_weights: c6_weights_v.as_slice(),
                        alpha_atomic_dynamic: alpha_dyn_v.as_slice(),
                        alpha_ct_dynamic: alpha_ct_dyn_v.as_deref(),
                        c6_iso: iso,
                        c6_aniso: c6_aniso_v.as_slice(),
                        c6_molecular_iso: c6_molecular_iso_v,
                    }),
                    _ => None,
                },
            },
        };
        // A file the caller asked for and did not get is not a warning.
        let write_failed = if let Err(e) = export_npz(npz_path, &npz_bundle) {
            eprintln!("error: failed to write {}: {}", npz_path, e);
            true
        } else {
            false
        };
        if !write_failed {
            println!("Wrote NPZ feature bundle: {}", npz_path);
            if c6_iso_opt.is_some() {
                println!(
                    "note: NPZ c6_iso/c6_aniso are per-atom PAIR tensors, not the \
                         molecular C6 total — do not sum them to approximate it (can be \
                         20-58% off). Read the NPZ key \"c6_molecular_iso\" for the \
                         correct DOSD-comparable value (project wiki: dosd-c6-rpa-vs-ts.md). \
                         The per-atom arrays are a PARTITION CONVENTION, not an \
                         observable; the NPZ keys \"c6_partition\"/\"c6_source\" record \
                         which one produced them (decode with .tobytes().decode())."
                );
            }
        }

        // Refuse to report success on a bundle that is missing something
        // the caller asked for.
        //
        // Each arm above warns and continues, which is often the right
        // trade -- the SCF is expensive and the properties that DID work
        // are worth keeping. What was wrong is that the PROCESS then exited
        // 0, making an incomplete NPZ indistinguishable from a complete one
        // to any caller that does not re-read stderr. A 500-molecule QM9
        // feature regeneration lost `alpha_atomic` on 476 of 500 molecules
        // that way, with 500 apparent successes.
        //
        // The file is still written before this check: a partial bundle is
        // salvageable, and deleting it would throw away work. Only the exit
        // status changes.
        let allow_partial = cfg.rpa.allow_partial_npz.unwrap_or(false);
        if (!npz_gaps.is_empty() || write_failed) && !allow_partial {
            eprintln!(
                "error: the NPZ bundle is incomplete — {} requested propert{} failed:",
                npz_gaps.len() + usize::from(write_failed),
                if npz_gaps.len() + usize::from(write_failed) == 1 {
                    "y"
                } else {
                    "ies"
                },
            );
            for g in &npz_gaps {
                eprintln!("  - {g}");
            }
            if write_failed {
                eprintln!("  - the bundle could not be written at all");
            }
            eprintln!(
                "Exiting nonzero so a caller does not mistake this for a complete \
                     bundle. Raise [memory] budget_gb if a gate refused, or set \
                     [rpa] allow_partial_npz = true to accept the gaps."
            );
            std::process::exit(1);
        }
    }
}

/// `method.kind = "gw"`. Extracted verbatim from the former `main()`
/// `"gw" => { ... }` match arm.
/// The open-shell reference for `method.kind = "gw"` and its label.
///
/// `[gw] reference` = `"uhf"` (default) runs `solve_uhf`, `"rohf"` runs
/// `solve_rohf` -- UKS/ROKS when `[rpa] xc` put a functional in
/// `rhf_config.xc` -- both with MOM after 5 DIIS iterations, the same
/// precedent as the `pdep-rpa` open-shell dispatch and the Python
/// `run_u_gw(reference=)`. `ferric_gw::run_u_gw` accepts either reference.
#[allow(clippy::too_many_arguments)]
fn gw_open_shell_reference(
    cfg: &Config,
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf_config: &RhfConfig,
) -> Result<(ferric_scf::result::ScfResult, &'static str), String> {
    let label = gw_open_shell_reference_label(cfg)?;
    let mut scf_cfg = rhf_config.clone();
    // MOM after 5 DIIS iters prevents orbital reordering on open-shell atoms.
    scf_cfg.mom_after_iter = 5;
    let result = match cfg.gw.parse_reference()? {
        config::GwReference::Uhf => solve_uhf(ctx, mol, prep, bounds, &scf_cfg),
        config::GwReference::Rohf => solve_rohf(ctx, mol, prep, op, bounds, &scf_cfg),
    };
    result
        .map(|r| (r, label))
        .map_err(|e| format!("({label} reference): {e}"))
}

/// The name of the open-shell GW reference `gw_open_shell_reference` solves:
/// UHF/ROHF, or UKS/ROKS when `[rpa] xc` is set.
fn gw_open_shell_reference_label(cfg: &Config) -> Result<&'static str, String> {
    let ks = cfg.rpa.xc.is_some();
    Ok(match cfg.gw.parse_reference()? {
        config::GwReference::Uhf => {
            if ks {
                "UKS"
            } else {
                "UHF"
            }
        }
        config::GwReference::Rohf => {
            if ks {
                "ROKS"
            } else {
                "ROHF"
            }
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn run_gw(
    cfg: &Config,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    op: Operator,
    result: &ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
) {
    let aux_name = cfg
        .rpa
        .auxbasis
        .as_deref()
        .unwrap_or(config::DEFAULT_CORRELATION_AUX);
    let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let scheme = cfg.rpa.parse_quadrature().unwrap_or_else(|e| {
        eprintln!("config error: {e}");
        std::process::exit(1);
    });
    let gw_method = cfg.gw.parse_method().unwrap_or_else(|e| {
        eprintln!("config error: {e}");
        std::process::exit(1);
    });
    // frozen_core must match between the PDEP (W) build and the GW self-
    // energy (Σ) build for self-consistency (see GwConfig::frozen_core
    // doc). [gw].frozen_core is the source of truth when set; otherwise
    // fall back to [rpa].frozen_core so a plain [rpa] block still works.
    let gw_frozen_core = cfg
        .gw
        .frozen_core
        .unwrap_or(cfg.rpa.frozen_core)
        .resolve(mol);
    let rpa_cfg = PdepRpaConfig {
        frozen_core: gw_frozen_core,
        trunc_thresh: cfg.rpa.trunc_thresh.unwrap_or(1e-4),
        eigensolver_max_vecs: 0,
        eigensolver_conv_thresh: cfg.rpa.eigensolver_conv_thresh.unwrap_or(1e-6),
        quadrature: QuadratureConfig {
            scheme,
            n_points: cfg.rpa.n_quad.unwrap_or(20),
            u0: cfg.rpa.u0.unwrap_or(0.5),
        },
        sternheimer: SternheimerConfig::default(),
        run_diagnostics: cfg.rpa.run_diagnostics,
        eigensolver: ferric_rpa::Eigensolver::default(),
        chi0_backend: ferric_rpa::config::Chi0Backend::default(),
        chi0_sparsity: cfg.rpa.parse_chi0_sparsity().unwrap_or_else(|e| {
            eprintln!("config error: {e}");
            std::process::exit(1);
        }),
        memory_budget_bytes: budget_bytes,
        // run_gw forces this on internally regardless of what's set
        // here (GW's Σ_c needs the inverse-dielectric stack), but set
        // it explicitly for clarity at the call site too.
        need_inv_dielectric_freq: true,
        need_eigenvalues_freq: true,
        verbose: cfg.scf.verbose,
    };
    let gw_cfg = ferric_gw::GwConfig {
        method: gw_method,
        qp_mos: cfg.gw.qp_mos.map(|[lo, hi]| lo..hi),
        max_ev_iter: cfg.gw.max_ev_iter.unwrap_or(20),
        ev_conv_thresh: cfg.gw.ev_conv_thresh.unwrap_or(1e-4),
        pade_npts: cfg.gw.pade_npts.unwrap_or(0),
        qp_newton_damp: cfg.gw.qp_newton_damp.unwrap_or(1.0),
        frozen_core: gw_frozen_core,
        memory_budget_bytes: budget_bytes,
        // Reuse the single CLI-wide `--verbose`/`-v` flag / `[scf]
        // verbose` TOML key rather than adding a parallel `[gw] verbose`.
        verbose: cfg.scf.verbose,
    };
    let ha_to_ev = 27.211_386_245_988_f64;
    if mol.multiplicity > 1 {
        // Open-shell path: `run()` already solved the reference through
        // `gw_open_shell_reference` (UHF, or ROHF for `[gw] reference =
        // "rohf"`; UKS/ROKS with `[rpa] xc`, MOM after 5 iterations), so
        // `result` is that reference; dispatch to run_u_gw.
        let ref_label = gw_open_shell_reference_label(cfg).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });
        // KS reference (RPA@PBE0-style): [rpa].xc set ⇒ `result` above is
        // already the UKS/ROKS solve; build per-spin vxc_diag so Σx−vxc enters
        // the per-spin QP equation inside run_u_gw. None (HF reference) ⇒ no
        // shift, matches run_u_gw's documented contract.
        // A ROHF/ROKS reference is semi-canonicalized (run_u_gw would do it
        // internally); doing it HERE lets the ROKS v_xc diagonal below be
        // evaluated on the same per-spin orbitals the QP equation uses. UHF/UKS
        // is borrowed unchanged.
        let result_u = ferric_scf::semicanonical::unrestricted_reference(mol, result)
            .unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
        let vxc_diag = match cfg.rpa.xc.as_deref() {
            Some(xc_name) => {
                let (diag_a, diag_b) = ferric_gw::vxc_mo::vxc_diagonal_mo(
                    mol, bs, xc_name, &result_u,
                )
                .unwrap_or_else(|e| {
                    eprintln!("error: vxc_diagonal_mo failed: {e}");
                    std::process::exit(1);
                });
                Some((diag_a, diag_b))
            }
            None => None,
        };
        let gw_result = ferric_gw::run_u_gw(
            mol,
            prep,
            &dfbs,
            op,
            &result_u,
            &rpa_cfg,
            &gw_cfg,
            vxc_diag.as_ref().map(|(a, b)| (a, b)),
        )
        .unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });
        println!(
            "U-GW[{:?}]/{} (aux: {}, ref: {ref_label}) on {}",
            gw_cfg.method, bs.name, aux_name, cfg.molecule.xyz
        );
        println!("  nbasis     = {}", prep.nbasis());
        println!("  {ref_label} energy: {:.10} Hartree", result.energy);
        println!("  ev iterations = {}", gw_result.n_ev_iter);
        println!("  outer converged = {}", gw_result.outer_converged);
        let two_s = mol.multiplicity as i64 - 1;
        let nocc_a = ((mol.nelec() as i64 + two_s) / 2) as usize;
        let nocc_b = ((mol.nelec() as i64 - two_s) / 2) as usize;
        for (spin_label, nocc, eps_mf, eps_qp, sigma_x, sigma_c, z_factor, qp_converged) in [
            (
                "alpha",
                nocc_a,
                &gw_result.eps_mf_a,
                &gw_result.eps_qp_a,
                &gw_result.sigma_x_a,
                &gw_result.sigma_c_a,
                &gw_result.z_factor_a,
                &gw_result.qp_converged_a,
            ),
            (
                "beta",
                nocc_b,
                &gw_result.eps_mf_b,
                &gw_result.eps_qp_b,
                &gw_result.sigma_x_b,
                &gw_result.sigma_c_b,
                &gw_result.z_factor_b,
                &gw_result.qp_converged_b,
            ),
        ] {
            println!("  -- {spin_label} spin channel --");
            println!(
                "  {:>4} {:>14} {:>14} {:>10} {:>10} {:>10}  qp_converged",
                "MO", "eps_mf(eV)", "eps_qp(eV)", "Sigma_x", "Sigma_c", "Z"
            );
            for (idx, &mo) in gw_result.mo_indices.iter().enumerate() {
                let tag = if nocc >= 1 && mo == nocc - 1 {
                    " (HOMO)"
                } else if mo == nocc {
                    " (LUMO)"
                } else {
                    ""
                };
                println!(
                    "  {:>4} {:>14.4} {:>14.4} {:>10.4} {:>10.4} {:>10.4}  {}{}",
                    mo,
                    eps_mf[idx] * ha_to_ev,
                    eps_qp[idx] * ha_to_ev,
                    sigma_x[idx],
                    sigma_c[idx],
                    z_factor[idx],
                    qp_converged[idx],
                    tag,
                );
            }
            if nocc >= 1 {
                if let Some(loc) = gw_result.mo_indices.iter().position(|&m| m == nocc - 1) {
                    println!("  {spin_label}-HOMO IP = {:.4} eV", -eps_qp[loc] * ha_to_ev);
                }
            }
            if let Some(loc) = gw_result.mo_indices.iter().position(|&m| m == nocc) {
                println!("  {spin_label}-LUMO EA = {:.4} eV", -eps_qp[loc] * ha_to_ev);
            }
        }
        if !gw_result.outer_converged {
            eprintln!(
                "warning: U-{:?} eigenvalue self-consistency did NOT converge in {} \
                     iterations (thresh {:.1e}); QP energies above are the last sweep",
                gw_cfg.method, gw_result.n_ev_iter, gw_cfg.ev_conv_thresh
            );
        }
        for (spin_label, flags) in [
            ("alpha", &gw_result.qp_converged_a),
            ("beta", &gw_result.qp_converged_b),
        ] {
            let unconverged_mos: Vec<usize> = gw_result
                .mo_indices
                .iter()
                .zip(flags.iter())
                .filter(|(_, &c)| !c)
                .map(|(&m, _)| m)
                .collect();
            if !unconverged_mos.is_empty() {
                eprintln!(
                    "warning: QP Newton solve did not converge for {spin_label} MO(s) \
                         {unconverged_mos:?}; those QP energies are best-effort"
                );
            }
        }
        return;
    }
    // KS reference (RPA@PBE0-style): [rpa].xc set ⇒ `result` above is
    // already the KS-DFT solve (via the xc/df_j_default/df_k_default
    // block); build vxc_diag so Σx−vxc enters the QP self-consistency.
    // None (HF reference) ⇒ no shift, matches run_gw's documented
    // contract.
    let vxc_diag = match cfg.rpa.xc.as_deref() {
        Some(xc_name) => {
            let (diag, _beta) = ferric_gw::vxc_mo::vxc_diagonal_mo(mol, bs, xc_name, result)
                .unwrap_or_else(|e| {
                    eprintln!("error: vxc_diagonal_mo failed: {e}");
                    std::process::exit(1);
                });
            Some(diag)
        }
        None => None,
    };
    let gw_result = ferric_gw::run_gw(
        mol,
        prep,
        &dfbs,
        op,
        result,
        &rpa_cfg,
        &gw_cfg,
        vxc_diag.as_ref(),
    )
    .unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let ref_label = if cfg.rpa.xc.is_some() { "KS" } else { "HF" };
    println!(
        "GW[{:?}]/{} (aux: {}, ref: {ref_label}) on {}",
        gw_cfg.method, bs.name, aux_name, cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!("  {ref_label} energy: {:.10} Hartree", result.energy);
    println!("  ev iterations = {}", gw_result.n_ev_iter);
    println!("  outer converged = {}", gw_result.outer_converged);
    println!(
        "  {:>4} {:>14} {:>14} {:>10} {:>10} {:>10}  qp_converged",
        "MO", "eps_mf(eV)", "eps_qp(eV)", "Sigma_x", "Sigma_c", "Z"
    );
    let nocc = (mol.nelec() as usize) / 2;
    for (idx, &mo) in gw_result.mo_indices.iter().enumerate() {
        let tag = if mo == nocc - 1 {
            " (HOMO)"
        } else if mo == nocc {
            " (LUMO)"
        } else {
            ""
        };
        println!(
            "  {:>4} {:>14.4} {:>14.4} {:>10.4} {:>10.4} {:>10.4}  {}{}",
            mo,
            gw_result.eps_mf[idx] * ha_to_ev,
            gw_result.eps_qp[idx] * ha_to_ev,
            gw_result.sigma_x[idx],
            gw_result.sigma_c[idx],
            gw_result.z_factor[idx],
            gw_result.qp_converged[idx],
            tag,
        );
    }
    if nocc >= 1 {
        if let Some(loc) = gw_result.mo_indices.iter().position(|&m| m == nocc - 1) {
            println!("  HOMO IP = {:.4} eV", -gw_result.eps_qp[loc] * ha_to_ev);
        }
    }
    if let Some(loc) = gw_result.mo_indices.iter().position(|&m| m == nocc) {
        println!("  LUMO EA = {:.4} eV", -gw_result.eps_qp[loc] * ha_to_ev);
    }
    if !gw_result.outer_converged {
        eprintln!(
            "warning: {:?} eigenvalue self-consistency did NOT converge in {} \
                 iterations (thresh {:.1e}); QP energies above are the last sweep",
            gw_cfg.method, gw_result.n_ev_iter, gw_cfg.ev_conv_thresh
        );
    }
    let unconverged_mos: Vec<usize> = gw_result
        .mo_indices
        .iter()
        .zip(gw_result.qp_converged.iter())
        .filter(|(_, &c)| !c)
        .map(|(&m, _)| m)
        .collect();
    if !unconverged_mos.is_empty() {
        eprintln!(
            "warning: QP Newton solve did not converge for MO(s) {unconverged_mos:?}; \
                 those QP energies are best-effort"
        );
    }
}

/// `method.kind = "bse-tda"`. Extracted verbatim from the former `main()`
/// `"bse-tda" => { ... }` match arm.
#[allow(clippy::too_many_arguments)]
fn run_bse_tda(
    cfg: &Config,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    op: Operator,
    result: &ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
) {
    // Closed-shell (RHF) only — run_bse_tda itself hard-errors on a
    // non-restricted reference; the top-level `result` above is always
    // an RHF solve for method.kind = "bse-tda" (no UHF branch, unlike
    // "gw"), so surface a clearer CLI-level message before the library
    // guard would otherwise fire.
    if mol.multiplicity > 1 {
        eprintln!(
            "error: method.kind = \"bse-tda\" is closed-shell (RHF) only; \
                 mol.multiplicity = {} is unsupported (no open-shell BSE-TDA exists)",
            mol.multiplicity
        );
        std::process::exit(1);
    }
    let aux_name = cfg
        .rpa
        .auxbasis
        .as_deref()
        .unwrap_or(config::DEFAULT_CORRELATION_AUX);
    let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let scheme = cfg.rpa.parse_quadrature().unwrap_or_else(|e| {
        eprintln!("config error: {e}");
        std::process::exit(1);
    });
    // frozen_core must match between the PDEP (W) build and the BSE/GW
    // self-energy build for self-consistency, same as the "gw" arm.
    // [gw].frozen_core is the source of truth when set; otherwise fall
    // back to [rpa].frozen_core.
    let bse_frozen_core = cfg
        .gw
        .frozen_core
        .unwrap_or(cfg.rpa.frozen_core)
        .resolve(mol);
    let rpa_cfg = PdepRpaConfig {
        frozen_core: bse_frozen_core,
        trunc_thresh: cfg.rpa.trunc_thresh.unwrap_or(1e-4),
        eigensolver_max_vecs: 0,
        eigensolver_conv_thresh: cfg.rpa.eigensolver_conv_thresh.unwrap_or(1e-6),
        quadrature: QuadratureConfig {
            scheme,
            n_points: cfg.rpa.n_quad.unwrap_or(20),
            u0: cfg.rpa.u0.unwrap_or(0.5),
        },
        sternheimer: SternheimerConfig::default(),
        run_diagnostics: cfg.rpa.run_diagnostics,
        eigensolver: ferric_rpa::Eigensolver::default(),
        chi0_backend: ferric_rpa::config::Chi0Backend::default(),
        chi0_sparsity: cfg.rpa.parse_chi0_sparsity().unwrap_or_else(|e| {
            eprintln!("config error: {e}");
            std::process::exit(1);
        }),
        memory_budget_bytes: budget_bytes,
        // run_bse_tda runs GW internally, which forces this on regardless
        // of what's set here; set it explicitly for clarity at the call
        // site too (matches the "gw" arm).
        need_inv_dielectric_freq: true,
        need_eigenvalues_freq: true,
        verbose: cfg.scf.verbose,
    };
    let ha_to_ev = 27.211_386_245_988_f64;
    let bse = ferric_gw::bse::run_bse_tda(mol, prep, &dfbs, op, result, &rpa_cfg, bse_frozen_core)
        .unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });
    println!(
        "BSE-TDA[G0W0@HF]/{} (aux: {}) on {}",
        bs.name, aux_name, cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!("  RHF energy = {:.10} Hartree", result.energy);
    println!(
        "  nocc = {}  nvir = {}  ({} singlet states)",
        bse.nocc,
        bse.nvir,
        bse.omega.len()
    );
    println!("  {:>4} {:>12} {:>10}", "n", "Omega (eV)", "f_osc");
    for (n, (&om, &f)) in bse
        .omega
        .iter()
        .zip(bse.oscillator_strength.iter())
        .enumerate()
    {
        println!("  {:>4} {:>12.4} {:>10.5}", n + 1, om * ha_to_ev, f);
    }
    println!(
        "  lowest singlet excitation = {:.4} eV  (f = {:.5})",
        bse.lowest_ev(),
        bse.lowest_oscillator_strength()
    );
}

/// `method.kind = "tdhf-static-polarizability"`. Extracted verbatim from the
/// former `main()` `"tdhf-static-polarizability" => { ... }` match arm.
#[allow(clippy::too_many_arguments)]
fn run_tdhf_static_polarizability(
    cfg: &Config,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    op: Operator,
    result: &ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
) {
    // RPAx@KS static (omega=0) polarizability only. SCOPE: this method
    // is deliberately narrow -- static alpha, nothing else. Do not
    // extend this arm to surface C6/dynamic alpha(iw); docs/VALIDATION.md
    // records a validated negative result for that extension of this
    // exact kernel (C6 stays ~63% low regardless of gap, worse than
    // ferric's production dRPA/PDEP C6 pipeline). See
    // ferric_gw::bse::run_rpax_static_polarizability's doc comment.
    if mol.multiplicity > 1 {
        eprintln!(
            "error: method.kind = \"tdhf-static-polarizability\" is closed-shell only; \
                 mol.multiplicity = {} is unsupported",
            mol.multiplicity
        );
        std::process::exit(1);
    }
    // The path is wired and checked for a KS reference only; require
    // [rpa].xc explicitly rather than silently falling back to an HF
    // reference (see the xc-routing block's comment above for why the old
    // "HF is much worse" justification was retracted).
    if cfg.rpa.xc.is_none() {
        eprintln!(
            "error: method.kind = \"tdhf-static-polarizability\" requires [rpa] xc \
                 (e.g. xc = \"PBE\") -- this path is only wired and checked for a \
                 Kohn-Sham reference"
        );
        std::process::exit(1);
    }
    let aux_name = cfg
        .rpa
        .auxbasis
        .as_deref()
        .unwrap_or(config::DEFAULT_CORRELATION_AUX);
    let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let scheme = cfg.rpa.parse_quadrature().unwrap_or_else(|e| {
        eprintln!("config error: {e}");
        std::process::exit(1);
    });
    let frozen_core = cfg
        .gw
        .frozen_core
        .unwrap_or(cfg.rpa.frozen_core)
        .resolve(mol);
    let scissor = cfg.gw.scissor.unwrap_or(0.0);
    let rpa_cfg = PdepRpaConfig {
        frozen_core,
        trunc_thresh: cfg.rpa.trunc_thresh.unwrap_or(1e-4),
        eigensolver_max_vecs: 0,
        eigensolver_conv_thresh: cfg.rpa.eigensolver_conv_thresh.unwrap_or(1e-6),
        quadrature: QuadratureConfig {
            scheme,
            n_points: cfg.rpa.n_quad.unwrap_or(20),
            u0: cfg.rpa.u0.unwrap_or(0.5),
        },
        sternheimer: SternheimerConfig::default(),
        run_diagnostics: cfg.rpa.run_diagnostics,
        eigensolver: ferric_rpa::Eigensolver::default(),
        chi0_backend: ferric_rpa::config::Chi0Backend::default(),
        chi0_sparsity: cfg.rpa.parse_chi0_sparsity().unwrap_or_else(|e| {
            eprintln!("config error: {e}");
            std::process::exit(1);
        }),
        memory_budget_bytes: budget_bytes,
        // No GW self-energy build in this path (static screening
        // modes from run_pdep_rpa only) -- unlike "gw"/"bse-tda",
        // this does NOT need the inverse-dielectric frequency stack.
        need_inv_dielectric_freq: false,
        need_eigenvalues_freq: true,
        verbose: cfg.scf.verbose,
    };
    let res = ferric_gw::bse::run_rpax_static_polarizability(
        mol,
        prep,
        &dfbs,
        op,
        result,
        &rpa_cfg,
        frozen_core,
        scissor,
    )
    .unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    println!(
        "RPAx@KS[{}] static polarizability /{} (aux: {}) on {}",
        cfg.rpa.xc.as_deref().unwrap_or("?"),
        bs.name,
        aux_name,
        cfg.molecule.xyz
    );
    println!(
        "  NOTE: static polarizability only -- do not use for C6/dispersion \
             (known negative accuracy result, see site/src/reference/validation.md)"
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!("  KS energy  = {:.10} Hartree", result.energy);
    println!("  nocc = {}  nvir = {}", res.nocc, res.nvir);
    println!("  alpha tensor (a.u.):");
    for row in &res.tensor {
        println!("    {:>12.6} {:>12.6} {:>12.6}", row[0], row[1], row[2]);
    }
    println!("  alpha_iso (static) = {:.6} a.u.", res.iso);
}

/// The printed name of an open-shell SCF: `hf` ("UHF"/"ROHF") for plain
/// Hartree-Fock, or `ks[functional]` ("UKS[PBE]"/"ROKS[PBE]") when
/// `RhfConfig::xc` promoted it to Kohn-Sham.
fn open_shell_scf_label(hf: &str, ks: &str, rhf_config: &RhfConfig) -> String {
    match rhf_config.xc.as_deref() {
        None => hf.to_string(),
        Some(f) => format!("{ks}[{f}]"),
    }
}

/// `method.task = "energy"` for the UHF/UKS route: `kind = "uhf"` (UKS when
/// `[dft] functional` is set) and `kind = "ksdft"` on an open-shell molecule
/// (see `Config::dispatch_kind`).
#[allow(clippy::too_many_arguments)]
fn run_uhf(
    cfg: &Config,
    ctx: &ParallelContext,
    mol: &Molecule,
    bs: &BasisSet,
    op: Operator,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    rhf_config: &RhfConfig,
) {
    log_jk_path(rhf_config, true);
    let result = solve_uhf(ctx, mol, prep, bounds, rhf_config).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let s_ov = ferric_integrals::oneelectron::overlap(prep);
    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    let nocc_a = ((nelec + two_s) / 2) as usize;
    let nocc_b = ((nelec - two_s) / 2) as usize;
    let s_true = 0.5 * (nocc_a as f64 - nocc_b as f64);
    let s_ideal = s_true * (s_true + 1.0);
    let c_a = result.mos_a();
    let c_b = result.mos_b();
    let overlap_ab = c_a
        .slice(ndarray::s![.., ..nocc_a])
        .t()
        .dot(&s_ov)
        .dot(&c_b.slice(ndarray::s![.., ..nocc_b]));
    let sum_sq: f64 = overlap_ab.iter().map(|v| v * v).sum();
    let s2 = s_ideal + (nocc_b as f64) - sum_sq;
    println!(
        "{}/{} on {}",
        open_shell_scf_label("UHF", "UKS", rhf_config),
        bs.name,
        cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!(
        "  mult       = {} (nocc_a={}, nocc_b={})",
        mol.multiplicity, nocc_a, nocc_b
    );
    println!("  iterations = {}", result.iterations);
    println!("  converged  = {}", result.converged);
    let dispersion =
        dispersion_correction(cfg, ctx, mol, bs, op, rhf_config, result.density_total());
    print_scf_energy(result.energy, dispersion.as_ref());
    print_cosx_final(&result);
    log_scf_dispersion(result.energy, dispersion.as_ref());
    // Open-shell energy runs return before `run()`'s `run_end`, so the final
    // pass gets its own record (as the dispersion correction does).
    if let (Some(f), Some(rl)) = (result.cosx_final, ferric_scf::runlog::log()) {
        rl.note("cosx_final_pass", cosx_final_json(&f));
    }
    println!("  <S^2>      = {:.6} (ideal {:.6})", s2, s_ideal);
    // task == "optimize" is handled by the top-level dispatch above
    // (optimize_geometry_uhf), which returns before reaching here.
}

/// `method.kind = "rohf"`, `task = "energy"`: ROHF, or ROKS when `[dft]
/// functional` is set.
#[allow(clippy::too_many_arguments)]
fn run_rohf(
    cfg: &Config,
    ctx: &ParallelContext,
    mol: &Molecule,
    bs: &BasisSet,
    op: Operator,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    rhf_config: &RhfConfig,
) {
    log_jk_path(rhf_config, true);
    let result = solve_rohf(ctx, mol, prep, op, bounds, rhf_config).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    let nocc_open = two_s as usize;
    let nocc_double = ((nelec - two_s) / 2) as usize;
    let s_true = 0.5 * two_s as f64;
    let s_ideal = s_true * (s_true + 1.0);
    println!(
        "{}/{} on {}",
        open_shell_scf_label("ROHF", "ROKS", rhf_config),
        bs.name,
        cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!(
        "  mult       = {} (nocc_double={}, nocc_open={})",
        mol.multiplicity, nocc_double, nocc_open
    );
    println!("  iterations = {}", result.iterations);
    println!("  converged  = {}", result.converged);
    let dispersion =
        dispersion_correction(cfg, ctx, mol, bs, op, rhf_config, result.density_total());
    print_scf_energy(result.energy, dispersion.as_ref());
    log_scf_dispersion(result.energy, dispersion.as_ref());
    println!("  <S^2>      = {:.6} (exact by construction)", s_ideal);
    // task == "optimize" is handled by the top-level dispatch above
    // (optimize_geometry_rohf), which returns before reaching here.
}

/// `method.task = "frequencies"` — harmonic vibrational frequencies.
///
/// FD of the ANALYTIC gradient (6N gradient evaluations), mass-weighted, with
/// translations/rotations projected out. The reference is chosen from
/// `method.kind` the same way `run_optimize` does; `[dft] xc` promotes RHF/UHF/
/// ROHF to the corresponding KS variant automatically.
///
/// With `[dft] dispersion` (closed-shell KS only, as in `run_optimize`) the
/// Hessian is the central difference of the CORRECTED gradient
/// `g_KS + g_disp`, each evaluated from the SCF converged at that displaced
/// geometry (`harmonic_frequencies_with_scf_correction`); MBD@rsSCS's density
/// dependence enters through its exact (Z-vector-relaxed) gradient.
fn run_frequencies(
    method: &str,
    cfg: &Config,
    ctx: &ParallelContext,
    mol: &Molecule,
    bs: &BasisSet,
    op: Operator,
    rhf_config: &RhfConfig,
) {
    use ferric_scf::frequencies::{FrequencyConfig, FrequencyReference, HessianMethod};

    let reference = match method {
        "rhf" | "ksdft" => FrequencyReference::Rhf,
        "uhf" => FrequencyReference::Uhf,
        "rohf" => FrequencyReference::Rohf,
        other => {
            eprintln!(
                "error: method.task = \"frequencies\" supports method.kind = rhf, uhf, \
                 rohf or ksdft (got \"{other}\"). Correlated methods have no analytic \
                 gradient, and a frequency run needs 6N of them."
            );
            std::process::exit(1);
        }
    };

    let mut fcfg = FrequencyConfig {
        reference,
        ..Default::default()
    };
    if let Some(d) = cfg.frequencies.delta {
        if !(d.is_finite() && d > 0.0) {
            eprintln!("error: [frequencies] delta must be finite and > 0 (got {d})");
            std::process::exit(1);
        }
        fcfg.delta = d;
    }
    if let Some(h) = cfg.frequencies.hessian.as_deref() {
        fcfg.hessian = HessianMethod::parse_config_str(h).unwrap_or_else(|e| {
            eprintln!("error: [frequencies] hessian: {e}");
            std::process::exit(1);
        });
    }

    let (res, at_reference) = frequencies_maybe_dispersion(
        cfg, ctx, mol, bs, op, rhf_config, &fcfg,
    )
    .unwrap_or_else(|e| {
        eprintln!("error computing frequencies: {e}");
        std::process::exit(1);
    });

    println!("Harmonic frequencies/{} on {}", bs.name, cfg.molecule.xyz);
    print_frequency_energy(&res, at_reference.as_ref());
    println!("  Hessian           = {}", res.hessian_source.label());
    println!("  gradient evals    = {}", res.n_gradient_evaluations);
    println!("  linear molecule   = {}", res.is_linear);
    // The asymmetry is zero in exact arithmetic, so it is a direct read on
    // whether `delta` and the SCF thresholds are sane for this system. Print it
    // unconditionally rather than burying it -- a large value invalidates every
    // number above it.
    println!("  Hessian asymmetry = {:.3e} Hartree/Bohr^2", res.asymmetry);
    println!("\n  mode   frequency (cm^-1)");
    for (k, w) in res.frequencies.iter().enumerate() {
        let tag = if *w < 0.0 { "  (imaginary)" } else { "" };
        println!("  {:>4}   {:>16.2}{}", k + 1, w, tag);
    }
    println!(
        "\n  projected trans/rot (should be ~0): {:?}",
        res.trans_rot_frequencies
            .iter()
            .map(|v| (v * 100.0).round() / 100.0)
            .collect::<Vec<_>>()
    );
}

/// The frequency run itself: plain, or on the KS + `[dft] dispersion` surface
/// (`harmonic_frequencies_with_scf_correction`), returning the correction at
/// the UNDISPLACED geometry for the printout and the run log (the driver
/// evaluates the closure there first).
#[allow(clippy::type_complexity)]
fn frequencies_maybe_dispersion(
    cfg: &Config,
    ctx: &ParallelContext,
    mol: &Molecule,
    bs: &BasisSet,
    op: Operator,
    rhf_config: &RhfConfig,
    fcfg: &ferric_scf::frequencies::FrequencyConfig,
) -> Result<
    (
        ferric_scf::frequencies::FrequencyResult,
        Option<DispersionCorrection>,
    ),
    ferric_core::FerricError,
> {
    use ferric_scf::frequencies::{
        harmonic_frequencies, harmonic_frequencies_with_scf_correction, FrequencyReference,
    };
    if fcfg.reference != FrequencyReference::Rhf {
        let label = if fcfg.reference == FrequencyReference::Uhf {
            open_shell_scf_label("UHF", "UKS", rhf_config)
        } else {
            open_shell_scf_label("ROHF", "ROKS", rhf_config)
        };
        refuse_open_shell_dispersion_gradient(cfg, &label, "frequencies");
    }
    // Refuse an unsupported configuration (e.g. hessian = "analytic") before
    // the dispersion model is built (MBD@rsSCS solves free atoms), not after.
    if dispersion_request(cfg).is_some() {
        ferric_scf::frequencies::check_scf_correction_config(fcfg)?;
    }
    let Some(model) = dispersion_gradient_model(
        cfg,
        ctx,
        mol,
        bs,
        op,
        rhf_config,
        "frequencies",
        ferric_scf::Spin::Restricted,
    ) else {
        return harmonic_frequencies(ctx, mol, &bs.name, op, rhf_config, fcfg).map(|r| (r, None));
    };
    let mut at_reference: Option<DispersionCorrection> = None;
    let res = harmonic_frequencies_with_scf_correction(
        ctx,
        mol,
        &bs.name,
        op,
        rhf_config,
        fcfg,
        |m, scf| {
            let (c, g) = model.evaluate(ctx, m, bs, op, rhf_config, scf)?;
            let e = c.energy;
            at_reference.get_or_insert(c);
            Ok((e, Some(g)))
        },
    )?;
    Ok((res, at_reference))
}

/// The energy lines of a frequency printout. With `[dft] dispersion` they
/// carry the corrected total, the KS energy and the correction at the input
/// geometry, which also go to the JSON run log as a `dispersion` record.
fn print_frequency_energy(
    res: &ferric_scf::frequencies::FrequencyResult,
    at_reference: Option<&DispersionCorrection>,
) {
    match at_reference {
        None => println!("  energy            = {:.10} Hartree", res.energy),
        Some(d) => {
            let scf_energy = res.energy - res.correction_energy;
            println!(
                "  energy            = {:.10} Hartree (KS-DFT + {})",
                res.energy, d.model
            );
            println!("  E(KS-DFT)         = {scf_energy:.10} Hartree");
            match &d.mbd {
                None => println!(
                    "  E(D3BJ)           = {:+.10} Hartree [params: {}]",
                    d.energy, d.params
                ),
                Some((beta, _)) => println!(
                    "  E(MBD@rsSCS)      = {:+.10} Hartree [beta: {beta}, functional: {}]",
                    d.energy, d.params
                ),
            }
            println!(
                "  dispersion        = in the Hessian (central differences of the KS + \
                 dispersion analytic gradient)"
            );
            log_scf_dispersion(scf_energy, Some(d));
        }
    }
}

/// `task.method = "optimize"` dispatch. Extracted verbatim from the former
/// `main()` `if task == "optimize" { ... }` block; each `method` sub-arm below
/// is byte-for-byte the original match-arm body.
#[allow(clippy::too_many_arguments)]
fn run_optimize(
    method: &str,
    cfg: &Config,
    ctx: &ParallelContext,
    mol: &Molecule,
    bs: &BasisSet,
    op: Operator,
    rhf_config: &RhfConfig,
    budget_bytes: Option<usize>,
) {
    let opt_config = OptimizeConfig {
        max_steps: cfg.optimize.max_steps.unwrap_or(100),
        g_max_thresh: cfg.optimize.g_max_thresh.unwrap_or(4.5e-4),
        g_rms_thresh: cfg.optimize.g_rms_thresh.unwrap_or(3.0e-4),
        e_conv: cfg.optimize.e_conv.unwrap_or(1e-6),
        trust_radius: cfg.optimize.trust_radius.unwrap_or(0.1),
        coord_system: match cfg.optimize.coordinates.as_deref() {
            None => Default::default(),
            Some(s) => crate::config::parse_coord_system(s).unwrap_or_else(|e| {
                eprintln!("error in [optimize]: {e}");
                std::process::exit(1);
            }),
        },
    };
    match method {
        "rhf" | "ksdft" => {
            // Dispersion (D3(BJ) or MBD@rsSCS) as an ADDITIVE correction on
            // both halves. It is threaded as a closure rather than as a flag
            // inside `ferric-scf` so that crate stays free of any dispersion
            // model; see `optimize_geometry_with_scf_correction`, which hands
            // the closure the converged SCF at each geometry (MBD@rsSCS needs
            // its density for the Hirshfeld volumes).
            //
            // The energy and the gradient come from the SAME resolved
            // parameters, which is the property that makes optimizing on this
            // surface meaningful -- a mismatched pair converges to a geometry
            // that is a stationary point of neither.
            let model = dispersion_gradient_model(
                cfg,
                ctx,
                mol,
                bs,
                op,
                rhf_config,
                "optimize",
                ferric_scf::Spin::Restricted,
            );
            let opt_result = optimize_geometry_with_scf_correction(
                ctx,
                mol,
                &bs.name,
                op,
                rhf_config,
                &opt_config,
                |m, scf| match &model {
                    Some(d) => {
                        let (c, g) = d.evaluate(ctx, m, bs, op, rhf_config, scf)?;
                        Ok((c.energy, Some(g)))
                    }
                    None => Ok((0.0, None)),
                },
            )
            .unwrap_or_else(|e| {
                eprintln!("error during optimization: {e}");
                std::process::exit(1);
            });
            println!("\nFinal Optimized Geometry (Bohr):");
            for (i, atom) in opt_result.mol.atoms.iter().enumerate() {
                println!(
                    "  {:2} {:2} {:12.8} {:12.8} {:12.8}",
                    i, atom.symbol, atom.x, atom.y, atom.zpos
                );
            }
            println!("\nOptimization Result:");
            println!("  converged  = {}", opt_result.converged);
            println!("  steps      = {}", opt_result.steps);
            println!("  final E    = {:.10} Hartree", opt_result.energy);
        }
        "pdep-rpa" => {
            let aux_name = cfg
                .rpa
                .auxbasis
                .as_deref()
                .unwrap_or(config::DEFAULT_CORRELATION_AUX);
            let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            let scheme = cfg.rpa.parse_quadrature().unwrap_or_else(|e| {
                eprintln!("config error: {e}");
                std::process::exit(1);
            });
            let rpa_cfg = PdepRpaConfig {
                frozen_core: cfg.rpa.frozen_core.resolve(mol),
                trunc_thresh: cfg.rpa.trunc_thresh.unwrap_or(1e-4),
                eigensolver_max_vecs: 0,
                eigensolver_conv_thresh: cfg.rpa.eigensolver_conv_thresh.unwrap_or(1e-8),
                quadrature: QuadratureConfig {
                    scheme,
                    n_points: cfg.rpa.n_quad.unwrap_or(16),
                    u0: cfg.rpa.u0.unwrap_or(0.5),
                },
                sternheimer: SternheimerConfig::default(),
                run_diagnostics: false,
                eigensolver: ferric_rpa::Eigensolver::default(),
                chi0_backend: ferric_rpa::config::Chi0Backend::default(),
                chi0_sparsity: cfg.rpa.parse_chi0_sparsity().unwrap_or_else(|e| {
                    eprintln!("config error: {e}");
                    std::process::exit(1);
                }),
                memory_budget_bytes: budget_bytes,
                // CLI RPA optimize is energy/gradient only (M9 gate).
                need_inv_dielectric_freq: false,
                // Energy/gradient only: no consumer reads `eigenvalues_freq`
                // here, so skip the per-frequency diagonalization and take the
                // LU log-det path for the correlation energy.
                need_eigenvalues_freq: false,
                verbose: cfg.scf.verbose,
            };
            let h_fd = 5e-4;
            let opt_result = ferric_rpa::optimize::optimize_geometry_rpa(
                mol,
                bs,
                &aux_bs,
                op,
                &rpa_cfg,
                &opt_config,
                h_fd,
            )
            .unwrap_or_else(|e| {
                eprintln!("error during RPA optimization: {e}");
                std::process::exit(1);
            });
            println!("\nFinal Optimized Geometry (Bohr):");
            for (i, atom) in opt_result.mol.atoms.iter().enumerate() {
                println!(
                    "  {:2} {:2} {:12.8} {:12.8} {:12.8}",
                    i, atom.symbol, atom.x, atom.y, atom.zpos
                );
            }
            println!("\nRPA Optimization Result:");
            println!("  converged  = {}", opt_result.converged);
            println!("  steps      = {}", opt_result.steps);
            println!(
                "  final E    = {:.10} Hartree (RHF + RPA)",
                opt_result.energy
            );
        }
        "rimp2" => {
            let aux_name = cfg
                .mp2
                .auxbasis
                .as_deref()
                .unwrap_or(config::DEFAULT_CORRELATION_AUX);
            let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            let mp2_config = RiMp2Config {
                frozen_core: cfg.mp2.frozen_core.resolve(mol),
                memory_budget_bytes: budget_bytes,
                ..Default::default()
            };
            let opt_result = ferric_mp2::optimize::optimize_geometry_rimp2(
                mol,
                bs,
                &aux_bs,
                op,
                &mp2_config,
                &opt_config,
                rhf_config.external_potential.as_ref(),
            )
            .unwrap_or_else(|e| {
                eprintln!("error during RI-MP2 optimization: {e}");
                std::process::exit(1);
            });
            println!("\nFinal Optimized Geometry (Bohr):");
            for (i, atom) in opt_result.mol.atoms.iter().enumerate() {
                println!(
                    "  {:2} {:2} {:12.8} {:12.8} {:12.8}",
                    i, atom.symbol, atom.x, atom.y, atom.zpos
                );
            }
            println!("\nRI-MP2 Optimization Result:");
            println!("  converged  = {}", opt_result.converged);
            println!("  steps      = {}", opt_result.steps);
            println!(
                "  final E    = {:.10} Hartree (RHF + MP2)",
                opt_result.energy
            );
        }
        "uhf" => {
            // UKS when `rhf_config.xc` is set (`kind = "uhf"` + functional, or
            // `ksdft` on an open-shell molecule): `optimize_geometry_uhf_with_scf_correction` then
            // takes `ks_gradient_uks`.
            // `[dft] dispersion` is applied as on the closed-shell path: D3(BJ)
            // from the geometry, MBD@rsSCS from the UKS spin densities with the
            // unrestricted Z-vector relaxation term.
            let label = open_shell_scf_label("UHF", "UKS", rhf_config);
            let model = dispersion_gradient_model(
                cfg,
                ctx,
                mol,
                bs,
                op,
                rhf_config,
                "optimize",
                ferric_scf::Spin::Unrestricted,
            );
            let opt_result = optimize_geometry_uhf_with_scf_correction(
                ctx,
                mol,
                &bs.name,
                op,
                rhf_config,
                &opt_config,
                |m, scf| match &model {
                    Some(d) => {
                        let (c, g) = d.evaluate(ctx, m, bs, op, rhf_config, scf)?;
                        Ok((c.energy, Some(g)))
                    }
                    None => Ok((0.0, None)),
                },
            )
            .unwrap_or_else(|e| {
                eprintln!("error during {label} optimization: {e}");
                std::process::exit(1);
            });
            println!("\nFinal Optimized Geometry (Bohr):");
            for (i, atom) in opt_result.mol.atoms.iter().enumerate() {
                println!(
                    "  {:2} {:2} {:12.8} {:12.8} {:12.8}",
                    i, atom.symbol, atom.x, atom.y, atom.zpos
                );
            }
            println!("\n{label} Optimization Result:");
            println!("  converged  = {}", opt_result.converged);
            println!("  steps      = {}", opt_result.steps);
            println!("  final E    = {:.10} Hartree", opt_result.energy);
        }
        "rohf" => {
            // ROKS (`ks_gradient_roks`) when `[dft] functional` is set.
            // `[dft] dispersion` as on the UKS path: D3(BJ) from the geometry,
            // MBD@rsSCS from the ROKS spin densities with the ROKS Z-vector
            // relaxation term.
            let label = open_shell_scf_label("ROHF", "ROKS", rhf_config);
            let model = dispersion_gradient_model(
                cfg,
                ctx,
                mol,
                bs,
                op,
                rhf_config,
                "optimize",
                ferric_scf::Spin::RestrictedOpen,
            );
            let opt_result = optimize_geometry_rohf_with_scf_correction(
                ctx,
                mol,
                &bs.name,
                op,
                rhf_config,
                &opt_config,
                |m, scf| match &model {
                    Some(d) => {
                        let (c, g) = d.evaluate(ctx, m, bs, op, rhf_config, scf)?;
                        Ok((c.energy, Some(g)))
                    }
                    None => Ok((0.0, None)),
                },
            )
            .unwrap_or_else(|e| {
                eprintln!("error during {label} optimization: {e}");
                std::process::exit(1);
            });
            println!("\nFinal Optimized Geometry (Bohr):");
            for (i, atom) in opt_result.mol.atoms.iter().enumerate() {
                println!(
                    "  {:2} {:2} {:12.8} {:12.8} {:12.8}",
                    i, atom.symbol, atom.x, atom.y, atom.zpos
                );
            }
            println!("\n{label} Optimization Result:");
            println!("  converged  = {}", opt_result.converged);
            println!("  steps      = {}", opt_result.steps);
            println!("  final E    = {:.10} Hartree", opt_result.energy);
        }
        _ => {
            eprintln!("error: geometry optimization is currently only supported for method.kind = \"rhf\", \"ksdft\", \"uhf\", \"rohf\", \"pdep-rpa\", or \"rimp2\"");
            std::process::exit(1);
        }
    }
}

/// TDA/TDDFT on a functional with no complete f_xc kernel (meta-GGA, VV10,
/// range-separated) is refused by `run_tddft`; check it before the reference
/// SCF is spent on a run that cannot finish.
fn refuse_tddft_xc_without_kernel(cfg: &Config, method: &str) {
    let Some(xc_name) = cfg.tddft.xc.as_deref() else {
        return;
    };
    if !matches!(method, "tda" | "tddft") {
        return;
    }
    if let Err(e) = ferric_dft::lr_kernel::resolve_singlet_response_xc(xc_name, "[tddft] xc") {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// `[dft] dispersion` is refused where no dispersion gradient is threaded:
/// any open-shell frequency run (the frequency driver's correction hook is
/// closed-shell only). Such a run would report the Hessian of the
/// UNCORRECTED surface while the config asks for a corrected one. The
/// single-point energy (task = "energy") applies the correction on every
/// reference, and `optimize` is supported on RKS, UKS and ROKS.
fn refuse_open_shell_dispersion_gradient(cfg: &Config, label: &str, task: &str) {
    if cfg.dft.dispersion.is_some() {
        eprintln!(
            "error: [dft] dispersion is not supported with method.task = \"{task}\" on an \
             open-shell ({label}) reference: the dispersion gradient is only threaded through \
             the closed-shell frequency driver, so this run would report the Hessian of the \
             uncorrected surface. Use task = \"energy\" for a corrected {label} single point, \
             task = \"optimize\" (supported with the correction on every reference), or remove \
             the key."
        );
        std::process::exit(1);
    }
}

fn run_tddft_arm(
    cfg: &Config,
    mol: &Molecule,
    _bs: &BasisSet,
    prep: &PreparedBasis,
    result: &ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
    method: &str,
    rhf_config: &RhfConfig,
) {
    use ferric_tddft::{TddftConfig, TddftMethod};

    let auxbasis_name = cfg
        .mp2
        .auxbasis
        .as_deref()
        .unwrap_or(config::TDDFT_DEFAULT_AUX);
    let dfbs_basis = basis::bundled(auxbasis_name).unwrap_or_else(|_| {
        eprintln!("error: auxiliary basis '{auxbasis_name}' not found");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &dfbs_basis).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });

    let tddft_method = match method {
        "tda" => TddftMethod::Tda,
        "tddft" => TddftMethod::Casida,
        _ => unreachable!(),
    };

    // `[tddft] xc` is the functional the reference SCF was converged with
    // (`run()` routes it into `RhfConfig::xc`), and it selects BOTH the
    // exact-exchange fraction and the f_xc kernel inside `run_tddft`. The
    // kernel grid is the SCF's own main grid, so a `[dft] grid_prune` setting
    // reaches the response too. `[tddft] c_hf` overrides only the exact-
    // exchange fraction (refused by `run_tddft` without a functional).
    let config = TddftConfig {
        n_roots: cfg.tddft.n_roots,
        method: tddft_method,
        // `[memory] budget_gb` now reaches the dense (ia,jb) matrices; this
        // parameter used to be `_budget_bytes`, accepted and dropped.
        memory_budget_bytes: budget_bytes,
        xc: cfg.tddft.xc.clone(),
        grid: rhf_config.dft_grid.clone().unwrap_or_default(),
        c_hf_override: cfg.tddft.c_hf,
    };

    let r = ferric_tddft::run_tddft(mol, prep, &dfbs, result, &config).unwrap_or_else(|e| {
        eprintln!("error: TDDFT failed: {e}");
        std::process::exit(1);
    });
    let c_hf = r.c_hf;

    let ha_to_ev = 27.211_386_245_988;
    println!(
        "{:?} — {} roots (c_HF = {:.2}, f_xc kernel: {}):",
        tddft_method,
        config.n_roots,
        c_hf,
        if r.fxc_included {
            "included"
        } else {
            "none (HF reference)"
        }
    );
    println!(
        "  {:>5}  {:>12}  {:>10}  {:>10}",
        "Root", "Energy (Ha)", "eV", "f"
    );
    for (i, (&e, &f)) in r
        .excitation_energies
        .iter()
        .zip(&r.oscillator_strengths)
        .enumerate()
    {
        println!(
            "  {:>5}  {:>12.6}  {:>10.4}  {:>10.6}",
            i + 1,
            e,
            e * ha_to_ev,
            f
        );
    }
}

#[cfg(test)]
mod local_printout_tests {
    use super::{local_json, local_reference_lines, model_label, LocalPrint};

    /// With the opt-in reference OFF the printout must say so and must never
    /// show the library's NaN sentinel or a NaN difference.
    ///
    /// Fails if reverted to printing the raw reference unconditionally (NaN
    /// twice once the library default is off); a helper that formatted the
    /// raw NaN instead of branching on `None` fails the `!contains("NaN")`
    /// assert, and one that printed nothing fails the "not computed" assert.
    #[test]
    fn reference_off_prints_not_computed_and_no_nan() {
        let text = local_reference_lines(
            -0.2,
            None,
            "E_corr(canonical RI)",
            "threshold error",
            "note",
        )
        .join("\n");
        assert!(
            !text.contains("NaN"),
            "NaN leaked into the printout:\n{text}"
        );
        assert!(text.contains("not computed"), "{text}");
        assert!(
            text.contains("[local] reference = true"),
            "the printout must name the opt-in key:\n{text}"
        );
        assert!(!text.contains("threshold error"), "{text}");
    }

    /// With the reference ON the value and the difference are printed with
    /// the labels aligned on the `=` column.
    #[test]
    fn reference_on_prints_value_and_signed_difference() {
        let lines = local_reference_lines(
            -0.2,
            Some(-0.25),
            "E_corr(canonical RI)",
            "total error",
            "maps",
        );
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert_eq!(lines[0], "  E_corr(canonical RI)  = -0.2500000000 Ha");
        assert_eq!(lines[1], "  total error           = +5.000e-2 Ha (maps)");
    }

    /// The model line names the model: "(exact)" with no threshold, or the
    /// scheme, the threshold and the kept fraction. The run-log `local`
    /// component is `null` exactly for the exact method.
    #[test]
    fn model_line_and_log_state_the_model() {
        assert_eq!(model_label("dRPA", None), "dRPA (exact)");
        let l = LocalPrint {
            eps: 1e-4,
            keep_fraction: 0.023,
            integral_direct: false,
        };
        assert_eq!(
            model_label("dRPA", Some(&l)),
            "dRPA (local: amplitude threshold, eps = 1.0e-4; kept 2.30% of amplitudes)"
        );
        let d = LocalPrint {
            integral_direct: true,
            ..l
        };
        assert!(model_label("MP2", Some(&d)).contains("integral-direct, eps = 1.0e-4"));
        assert!(local_json(None).is_null());
        let j = local_json(Some(&d));
        assert_eq!(j["scheme"], "amplitude-threshold");
        assert_eq!(j["eps"], 1e-4);
        assert_eq!(j["keep_fraction"], 0.023);
        assert_eq!(j["integral_direct"], true);
    }
}
