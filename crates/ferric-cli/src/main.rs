mod config;

use config::{load_config, Config};
use ferric_core::basis;
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::att_vv10::{
    att_mp2_vv10, u_att_mp2_vv10, AttVv10Attenuator, AttVv10SpinComponents,
};
use ferric_mp2::attenuated::{attenuated_ri_mp2, AttenuatedMp2Config};
use ferric_mp2::laplace::{laplace_ri_mp2, laplace_sos_mp2, SosFormulation, SosMp2Config};
use ferric_mp2::mp3::mp3_energy;
use ferric_mp2::oo_rimp2::{oo_ri_mp2, OoRiMp2Config};
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_mp2::scs::{scs_mp2, scs_mp2_2terfc, ScsMp2Config, ScsMp2TerfcConfig};
use ferric_rpa::config::{QuadratureConfig, SternheimerConfig};
use ferric_rpa::{run_pdep_rpa, PdepRpaConfig};
use ferric_cc::ccsd::ccsd;
use ferric_cc::ccsd_closed_shell::ccsd_closed_shell;
use ferric_cc::double_hybrid::{run_wb97x_l_v, DoubleHybridConfig};
use ferric_cc::linlccd::{linlccd, LadderVariant};
use ferric_cc::CcConfig;
use ferric_core::parallel::ParallelContext;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::uhf::solve_uhf;
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::optimize::{optimize_geometry, optimize_geometry_rohf, optimize_geometry_uhf, OptimizeConfig};

/// Run `f` on a private single-thread rayon pool.
///
/// Free-atom / proatom SCFs are tiny (one atom, ~10-30 basis functions). On the
/// global multi-thread pool, rayon's per-task coordination overhead dwarfs the
/// actual Fock-build work — a single S atom at aug-cc-pVDZ took 179 s with
/// RAYON_NUM_THREADS=8 vs 9.6 s with 1 (18× slower). Since every TS volume and
/// Hirshfeld proatom triggers such a solve, the penalty made 2nd-row molecules
/// (h2s, hcl) take 40-60 min. Confining these inner solves to one thread keeps
/// the big molecular SCF/RPA fully parallel while making the atoms fast.
fn run_serial<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    match rayon::ThreadPoolBuilder::new().num_threads(1).build() {
        Ok(pool) => pool.install(f),
        Err(_) => f(), // if pool creation fails, just run inline
    }
}

fn print_usage() {
    eprintln!("usage: ferric [--verbose|-v] <input.toml>");
    eprintln!();
    eprintln!("Run a ferric quantum-chemistry calculation from a TOML input file.");
    eprintln!("See examples/*.toml for sample inputs and docs/quickstart.md for a walkthrough.");
    eprintln!();
    eprintln!("  --verbose, -v   Print one line per SCF iteration to stdout (energy, dE,");
    eprintln!("                  density/DIIS error) as the job runs. Same effect as setting");
    eprintln!("                  `verbose = true` in the [scf] TOML section.");
}

/// Epistemic-status warnings for `method.kind` values that are graded Smoke
/// or Stub in `docs/VALIDATION.md` (i.e. NOT Proven / Proven (narrow)).
///
/// SOURCE OF TRUTH: `docs/VALIDATION.md`. This table is a condensed,
/// CLI-facing pointer into it, not a second grading system -- when a
/// method's grade in VALIDATION.md changes (promoted to Proven, demoted to
/// Stub, caveat text edited), update BOTH this table and the doc. Proven /
/// Proven (narrow) methods (rhf, uhf, rohf, ksdft, rimp2, mp3, att-rimp2,
/// scs-mp2, scs-mp2-2terfc, laplace-mp2, pdep-rpa, ccsd, linlccd) do not
/// appear here and never print a warning.
const EPISTEMIC_WARNINGS: &[(&str, &str)] = &[
    (
        "gw",
        "method.kind = \"gw\" is Smoke-grade (see docs/VALIDATION.md): G0W0/COHSEX/evGW0/evGW \
         validated to ~5 meV vs MOLGW on a single H2O/cc-pVDZ case but most asserts are loose \
         range bands; treat results as accurate to roughly +/-0.3 eV, not a reference number.",
    ),
    (
        "bse-tda",
        "method.kind = \"bse-tda\" is Smoke-grade (see docs/VALIDATION.md): only excitation \
         ordering and a physicality gate are checked; the excitation-energy gap error is \
         inherited directly from the underlying GW quasiparticle gap (also Smoke-grade).",
    ),
    (
        "tdhf-static-polarizability",
        "method.kind = \"tdhf-static-polarizability\" is Smoke-grade (see docs/VALIDATION.md): \
         static alpha at RPAx@KS matches DOSD water closely in the one case checked, but the \
         same dense TDHF/RPAx kernel gives C6 ~63% low regardless of gap -- do not extrapolate \
         this method's accuracy beyond static alpha on a KS reference.",
    ),
    (
        "rs-mp2-rpa",
        "method.kind = \"rs-mp2-rpa\" has Proven energy LIMITS (omega->0/infinity reduce exactly \
         to MP2/MP2+dRPA) but is only Smoke-grade at production omega (see docs/VALIDATION.md): \
         ACONF ties RI-MP2 at omega<=0.3 1/A, and the aug-cc-pVTZ benchmark criterion was met \
         only marginally on one small subset -- treat mid-range-omega numbers as unproven on \
         new systems.",
    ),
    (
        "mp2-v",
        "method.kind = \"mp2-v\" is Smoke-grade (see docs/VALIDATION.md): the VV10 half is proven \
         bit-identical to the wB97X-V code path and the damping is validated by limits, but there \
         is NO comparison to any published MP2-V number (the paper reports only S66/G2 statistics, \
         never a total energy). The defaults (r0 = 1.00 A, b = 11.0, C = 0.0089, terfc, post-HF) \
         are fitted for aug-cc-pVTZ, no counterpoise, frozen core -- running another basis, or \
         with [mp2] frozen_core = 0 (the default here), is unparameterized extrapolation. \
         Open-shell (multiplicity > 1) is DOUBLY unvalidated: S66 is entirely closed-shell, so no \
         open-shell parameterization exists at all.",
    ),
    (
        "oo-rimp2",
        "method.kind = \"oo-rimp2\" is Smoke-grade (see docs/VALIDATION.md): orbital \
         optimization is checked for internal self-consistency (converged stationary point, \
         analytic gradient vanishes) but there is NO external absolute-energy reference -- \
         PySCF/psi4/forte all lack a directly comparable OO-MP2 implementation.",
    ),
    (
        "wb97x-l-v",
        "method.kind = \"wb97x-l-v\" is Smoke-grade (see docs/VALIDATION.md): the functional runs \
         end to end and its pieces (E_KS, E_c, the lambda scaling, the omega range separation) are \
         separately checked against the paper's structure and limits, but NO reference value for \
         the TOTAL energy exists in ferric -- nothing compares it to the paper or to another code. \
         Do not quote a wB97X-L-V total energy as validated.",
    ),
];

/// Print a one-line epistemic-status warning to stderr if `method` is a
/// Smoke/Stub-grade `method.kind` per `docs/VALIDATION.md`. No-op (and no
/// output) for Proven / Proven (narrow) methods.
fn warn_if_epistemically_unproven(method: &str) {
    if let Some((_, text)) = EPISTEMIC_WARNINGS.iter().find(|(k, _)| *k == method) {
        eprintln!("[warning] {text}");
    }
}

pub fn main() {
    // Safe-by-default threading: pin OpenBLAS to 1 thread (rayon owns ferric's
    // parallelism) unless the user explicitly set OPENBLAS_NUM_THREADS. Without
    // this, running the release binary directly oversubscribes rayon × BLAS.
    ferric_integrals::blas_threads::init_threading();
    let ctx = ParallelContext::new();
    let args: Vec<String> = std::env::args().collect();
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
    for arg in &args[1..] {
        match arg.as_str() {
            "--verbose" | "-v" => cli_verbose = true,
            other if toml_path.is_none() => toml_path = Some(other),
            _ => {
                eprintln!("usage: ferric [--verbose|-v] <input.toml>");
                std::process::exit(2);
            }
        }
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
    let method = cfg.method.kind.as_str();
    let task = cfg.method.task.as_str();
    if !matches!(method, "rhf" | "uhf" | "rohf" | "ksdft" | "rimp2" | "mp3" | "oo-rimp2" | "att-rimp2" | "mp2-v" | "scs-mp2" | "scs-mp2-2terfc" | "laplace-mp2" | "laplace-sos-mp2" | "pdep-rpa" | "rs-mp2-rpa" | "gw" | "bse-tda" | "tdhf-static-polarizability" | "ccsd" | "linlccd" | "wb97x-l-v") {
        eprintln!("error: unsupported method.kind = \"{method}\"; expected rhf, uhf, rohf, ksdft, rimp2, mp3, oo-rimp2, att-rimp2, mp2-v, scs-mp2, scs-mp2-2terfc, laplace-mp2, laplace-sos-mp2, pdep-rpa, rs-mp2-rpa, gw, bse-tda, tdhf-static-polarizability, ccsd, linlccd, or wb97x-l-v");
        std::process::exit(1);
    }
    warn_if_epistemically_unproven(method);
    if !matches!(task, "energy" | "optimize") {
        eprintln!("error: unsupported method.task = \"{task}\"; expected energy or optimize");
        std::process::exit(1);
    }
    let mut mol = Molecule::load_xyz_with_charge(&cfg.molecule.xyz, cfg.molecule.charge, cfg.molecule.multiplicity).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
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

    let prep = PreparedBasis::new(&mol, &bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    // For ksdft, default RI-J/RI-K to def2-universal-jkfit (required for hybrids
    // and RSH; harmless for pure DFT). User can still override via [scf].
    let (xc, df_j_default, df_k_default) = if method == "ksdft" {
        let functional = cfg.dft.functional.clone().unwrap_or_else(|| "LDA".into());
        (
            Some(functional),
            Some("def2-universal-jkfit".to_string()),
            Some("def2-universal-jkfit".to_string()),
        )
    } else if matches!(method, "pdep-rpa" | "rpa" | "gw" | "tdhf-static-polarizability") && cfg.rpa.xc.is_some() {
        // RPA on a KS-DFT reference (RPA@PBE0 etc.): run the closed-shell KS
        // solver for the reference orbitals. Hybrids need RI-J/RI-K.
        // GW reuses [rpa].xc for its own KS-reference switch (GW needs the
        // same vxc_diag plumbing pdep-rpa's KS path already has). BSE-TDA is
        // closed-shell (RHF) only (see the "bse-tda" arm's guard) and does
        // not currently expose a KS-reference switch, so it is intentionally
        // excluded from this branch even though it's included below.
        // "tdhf-static-polarizability" (RPAx@KS static alpha) REQUIRES a KS
        // reference -- its own dispatch arm hard-errors below if [rpa].xc is
        // unset, since the validated static-alpha accuracy (9.24 vs DOSD
        // 9.64 a.u. on water/PBE) is a KS-reference result; the HF-reference
        // variant of this same kernel gives a much worse static alpha (5.24
        // a.u. gate-2 measurement) and is deliberately not offered here.
        (
            cfg.rpa.xc.clone(),
            Some("def2-universal-jkfit".to_string()),
            Some("def2-universal-jkfit".to_string()),
        )
    } else if matches!(method, "pdep-rpa" | "rpa" | "rs-mp2-rpa" | "gw" | "bse-tda" | "tdhf-static-polarizability") {
        // RPA@HF (no xc): the HF reference SCF defaults to RI-J/RI-K with
        // def2-universal-jkfit too. Exact 4-index J/K per iteration makes the
        // HF reference 10-20× slower than the RI-JK PBE reference (hcl/aug-cc-
        // pVTZ: 505 s vs 25 s) for no benefit — the RI-JK fitting error (~µHa)
        // is far below the C6 differences we study. Keep SCF aux separate from
        // the RPA correlation aux / SR-MP2+LR-RPA correlation aux
        // (see ferric-jk-aux-convention).
        (
            None,
            Some("def2-universal-jkfit".to_string()),
            Some("def2-universal-jkfit".to_string()),
        )
    } else {
        (None, None, None)
    };
    // Unified memory budget from [memory] (bytes), threaded into EVERY method
    // config below. `None` → each method's resolver auto-detects (0.8 × RAM).
    // Log the resolved value + source once, up front, so runs are auditable.
    let budget_bytes: Option<usize> = cfg.memory.budget_bytes();
    {
        let resolution = ferric_core::memory::resolve_budget(budget_bytes);
        eprintln!("[ferric] {}", resolution.audit_line());
    }
    let rhf_config = RhfConfig {
        max_iter: cfg.scf.max_iter,
        energy_conv: cfg.scf.energy_conv,
        density_conv: cfg.scf.density_conv,
        diis_size: cfg.scf.diis_size,
        diis_flavor: cfg.scf.diis_flavor(),
        diis_switch_thresh: cfg.scf.diis_switch_thresh.unwrap_or(1e-1),
        smearing_sigma: cfg.scf.smearing_sigma,
        integral_thresh: cfg.scf.integral_thresh,
        k_builder: cfg.scf.k_builder.clone(),
        df_j_aux: cfg.scf.df_j_aux.clone().or(df_j_default),
        df_k_aux: cfg.scf.df_k_aux.clone().or(df_k_default),
        xc,
        dft_grid: None,
        nlc_grid: None,
        level_shift: cfg.scf.level_shift.unwrap_or(0.0),
        newton_trigger: if cfg.scf.soscf { 1e-3 } else { 0.0 },
        ah_trigger: 0.0,
        mom_after_iter: cfg.scf.mom_after_iter,
        constraints: Vec::new(),
        cdft_lambda_tol: 1e-5,
        fractional_occ: false,
        // 0 = "unset" → the SCF resolver auto-detects (0.8×RAM). An explicit
        // [memory] budget (incl. a deliberate 2 GiB) is passed through and honored.
        three_index_budget_bytes: budget_bytes.unwrap_or(0),
        init_guess_density: None,
        use_sad_guess: cfg.scf.use_density_guess(),
        stall_window: None,
        divergence_tol: None,
        external_potential: cfg.external_potential.to_external_potential(),
        cosmo: cfg.cosmo.clone(),
        // TODO(pcm-cli-wiring): no [pcm] TOML section yet -- PCM is only
        // reachable via the ferric-scf/ferric-python APIs for now. Wiring a
        // CLI-level PcmConfig (mirroring the [external_potential] section)
        // is a natural follow-up, out of scope for the initial PCM landing.
        pcm: None,
        verbose: cfg.scf.verbose,
    };

    if task == "optimize" {
        run_optimize(method, &cfg, &ctx, &mol, &bs, op, &rhf_config, budget_bytes);
        return;
    }

    if method == "uhf" {
        run_uhf(&cfg, &ctx, &mol, &bs, &prep, &bounds, &rhf_config);
        return;
    }

    if method == "rohf" {
        run_rohf(&cfg, &ctx, &mol, &bs, op, &prep, &bounds, &rhf_config);
        return;
    }

    // Both new correlated methods are closed-shell (RHF-reference) only: they
    // reach `eps_r()`/`mos_r()`, which assert on `Spin::Restricted`. Reject an
    // open-shell request HERE, before the shared `solve_rhf` below, because that
    // solve fails first on an odd electron count and reports the misleading
    // "SCF did not converge after 0 iterations" rather than the real reason.
    if matches!(method, "linlccd" | "wb97x-l-v") && mol.multiplicity > 1 {
        eprintln!(
            "error: method.kind = \"{method}\" requires a closed-shell (restricted) reference; \
             open-shell LinLCCD(hh) / wB97X-L-V are library-only \
             (ferric_cc::linlccd_u::u_linlccd, \
             ferric_cc::double_hybrid::u_solve_wb97x_l_v)"
        );
        std::process::exit(1);
    }

    // The wB97X-L-V double hybrid converges its OWN Kohn-Sham reference inside
    // `run_wb97x_l_v` (it forces `xc = "wB97X-L-V"` and runs `ksdft_ladder`), so
    // it returns early here rather than falling through to the unconditional
    // `solve_rhf` below. Letting it fall through would run a full plain-HF SCF
    // whose result is then thrown away, and — worse — the reference actually
    // consumed would silently be the wrong one.
    if method == "wb97x-l-v" {
        run_wb97x_l_v_arm(&cfg, &ctx, &mol, &bs, &prep, &bounds, &rhf_config, budget_bytes);
        return;
    }

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
        let ladder = cfg.scf.build_ladder(&rhf_config);
        // Report the J/K path actually in use. RI-JK is now opt-in (the ladder
        // no longer substitutes it — see `ladder::default_ladder_from`), but
        // KS-DFT still auto-selects an aux above, so state which one ran rather
        // than leaving a CLI-vs-library energy comparison to guesswork.
        if let Some(rung0) = ladder.first() {
            match rung0.config.df_j_aux.as_deref() {
                Some(aux) => eprintln!("[ferric] SCF J/K: RI-JK via {aux}"),
                None => eprintln!("[ferric] SCF J/K: exact 4-index (set [scf] df_j_aux/df_k_aux for RI-JK)"),
            }
        }
        let lr = ferric_scf::ladder::solve_rhf_ladder(&ctx, &mol, &prep, op, &bounds, &ladder)
            .unwrap_or_else(|e| { eprintln!("error: SCF ladder failed: {e:?}"); std::process::exit(1); });
        if !lr.converged {
            eprintln!("warning: SCF did not fully converge (best rung {}, exit {:?})", lr.rung_reached, lr.rung_outcomes.last().map(|o| o.exit));
        }
        lr.result
    } else {
        solve_rhf(&ctx, &mol, &prep, op, &bounds, &rhf_config).unwrap_or_else(|e| {
        // For pdep-rpa/gw/mp2-v with open-shell molecules the UHF dispatch inside
        // the arm handles convergence; the global RHF result is not used.
        // (mp2-v: `run_mp2_v` dispatches on `result.spin`, so the UHF result
        // produced here IS what it consumes — it does not re-solve.)
        if (method == "pdep-rpa" || method == "gw" || method == "mp2-v") && mol.multiplicity > 1 {
            // Return a dummy result — it will be shadowed immediately in the arm.
            // The SCF failure is expected here; suppress the exit.
            let _ = e;
            // We cannot construct a valid ScfResult without running SCF.
            // Fall back: run UHF here so `result` is valid even if the arm
            // never uses it (e.g. if the match falls through to _ => unreachable!).
            solve_uhf(&ctx, &mol, &prep, &bounds, &{
                let mut c = rhf_config.clone(); c.mom_after_iter = 5; c
            }).unwrap_or_else(|e2| {
                eprintln!("error (pre-UHF): {e2}");
                std::process::exit(1);
            })
        } else {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
        })
    };

    // Ad-hoc same-basis Hirshfeld proatom: neutral free-atom densities computed
    // in the molecule's OWN basis (basis-consistent partition; fixes the legacy
    // single-Slater H-starvation). Built lazily via atomic SCF; shared by all
    // Hirshfeld consumers (charges, effective volumes, per-atom polarizability).
    let proatom_radii: Vec<f64> = (1..=600).map(|k| k as f64 * 0.05).collect(); // 0.05..30 Bohr
    let proatom_gs_mult = |z: i32| -> usize {
        match z {
            // Doublets: H, Li, B, F, Na, Al, Cl, Ga, Br (one unpaired p/s e⁻)
            1 | 3 | 5 | 9 | 11 | 13 | 17 | 31 | 35 | 53 => 2,
            // ²S alkali-like heavy atoms + coinage metals (single ns valence e⁻):
            // K, Cu, Rb, Ag. Kept in sync with guess::atom_ground_state_mult —
            // without these an odd-electron atom hits `_ => 1` and its closed-shell
            // proatom RHF fails at iter 0 (breaks the Hirshfeld/TS proatom for any
            // Cu/K/Rb/Ag-containing molecule).
            19 | 29 | 37 | 47 => 2,
            // Triplets (³P): C, O, Si, S, Ge, Se
            6 | 8 | 14 | 16 | 32 | 34 => 3,
            // Quartets (⁴S): N, P, As
            7 | 15 | 33 => 4,
            // Odd electron count can never be a singlet: default odd Z to a doublet.
            _ if z % 2 == 1 => 2,
            _ => 1,
        }
    };
    let proatom = |z: i32, qi: i32| -> Option<ferric_rpa::properties::RadialProatom> {
        if qi != 0 || z - qi <= 0 {
            return None; // neutral only; ions via fallback
        }
        let sym = ferric_core::elements::z_to_symbol(z).unwrap_or("X");
        let axyz = format!("1\n{sym}\n{sym} 0 0 0\n");
        let amol = Molecule::parse_xyz(&axyz, 0, proatom_gs_mult(z)).ok()?;
        let aobs = PreparedBasis::new(&amol, &bs).ok()?;
        let abounds = SchwarzBounds::compute(op, &aobs).ok()?;
        let mut acfg = rhf_config.clone();
        // Run the single-atom SCF on a 1-thread pool — see run_serial.
        let adens = run_serial(|| {
            if proatom_gs_mult(z) == 1 {
                solve_rhf(&ctx, &amol, &aobs, op, &abounds, &acfg)
                    .ok()
                    .map(|r| r.density_r().to_owned())
            } else {
                acfg.mom_after_iter = 5;
                // KS-DFT free-atom solve: fractional/ensemble occupation spreads
                // the open-shell electrons equally over degenerate frontier
                // orbitals (e.g. Br 4p⁵ ²P, O/S 2p³ ³P), restoring spherical
                // symmetry so the GGA XC potential doesn't oscillate. Pure HF
                // free-atom solves don't suffer this (K is orbital-invariant in
                // the degenerate subspace), so only enable when xc is set.
                if acfg.xc.is_some() {
                    acfg.fractional_occ = true;
                }
                solve_uhf(&ctx, &amol, &aobs, &abounds, &acfg)
                    .ok()
                    .map(|r| r.density_total().to_owned())
            }
        })?;
        ferric_rpa::properties::spherically_averaged_proatom(z, &bs, &adens, &proatom_radii).ok()
    };

    match method {
        "rhf" => run_rhf(&cfg, &bs, &prep, &result),
        "ksdft" => run_ksdft(&cfg, &bs, &prep, &result),
        "rimp2" => run_rimp2(&cfg, &mol, &bs, &prep, op, &result, budget_bytes),
        "mp3" => run_mp3(&cfg, &mol, &bs, &prep, op, &result),
        "oo-rimp2" => run_oo_rimp2(&cfg, &mol, &bs, &prep, op, &bounds, &result, budget_bytes),
        "att-rimp2" => run_att_rimp2(&cfg, &mol, &bs, &prep, &result, budget_bytes),
        "mp2-v" => run_mp2_v(&cfg, &mol, &bs, &prep, &result, budget_bytes),
        "rs-mp2-rpa" => run_rs_mp2_rpa(&cfg, &mol, &bs, &prep, &result, budget_bytes),
        "scs-mp2" => run_scs_mp2(&cfg, &mol, &bs, &prep, &result, budget_bytes),
        "scs-mp2-2terfc" => run_scs_mp2_2terfc(&cfg, &mol, &bs, &prep, &result, budget_bytes),
        "ccsd" => run_ccsd(&cfg, &mol, &bs, &prep, op, &result, budget_bytes),
        "linlccd" => run_linlccd(&cfg, &mol, &bs, &prep, op, &result, budget_bytes),
        "laplace-mp2" => run_laplace_mp2(&cfg, &mol, &bs, &prep, op, &result),
        "laplace-sos-mp2" => {
            run_laplace_sos_mp2(&cfg, &mol, &bs, &prep, op, &result, budget_bytes)
        }
        "pdep-rpa" => run_pdep_rpa_arm(
            &cfg, &ctx, &mol, &bs, &prep, op, &bounds, &rhf_config, result, budget_bytes,
            &proatom_gs_mult, &proatom,
        ),
        "gw" => run_gw(&cfg, &ctx, &mol, &bs, &prep, op, &bounds, &rhf_config, &result, budget_bytes),
        "bse-tda" => run_bse_tda(&cfg, &mol, &bs, &prep, op, &result, budget_bytes),
        "tdhf-static-polarizability" => run_tdhf_static_polarizability(&cfg, &mol, &bs, &prep, op, &result, budget_bytes),
        _ => unreachable!(),
    }
}

/// `method.kind = "rhf"`. Extracted verbatim from the former `main()`
/// `"rhf" => { ... }` match arm.
fn run_rhf(cfg: &Config, bs: &BasisSet, prep: &PreparedBasis, result: &ferric_scf::result::ScfResult) {
    println!("RHF/{} on {}", bs.name, cfg.molecule.xyz);
    println!("  nbasis     = {}", prep.nbasis());
    println!("  iterations = {}", result.iterations);
    println!("  converged  = {}", result.converged);
    println!("  energy     = {:.10} Hartree", result.energy);
}

/// `method.kind = "ksdft"`. Extracted verbatim from the former `main()`
/// `"ksdft" => { ... }` match arm.
fn run_ksdft(cfg: &Config, bs: &BasisSet, prep: &PreparedBasis, result: &ferric_scf::result::ScfResult) {
    let functional = cfg.dft.functional.as_deref().unwrap_or("LDA");
    println!("KS-DFT[{functional}]/{} on {}", bs.name, cfg.molecule.xyz);
    println!("  nbasis     = {}", prep.nbasis());
    println!("  iterations = {}", result.iterations);
    println!("  converged  = {}", result.converged);
    println!("  energy     = {:.10} Hartree", result.energy);
}

/// `method.kind = "rimp2"`. Extracted verbatim from the former `main()`
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
    let aux_name = cfg.mp2.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
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
            frozen_core: cfg.mp2.frozen_core,
            memory_budget_bytes: budget_bytes,
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    println!(
        "RI-MP2/{} (aux: {}) on {}",
        bs.name, aux_name, cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!("  RHF energy = {:.10} Hartree", result.energy);
    println!("  MP2 corr   = {:.10} Hartree", mp2_result.mp2_corr);
    println!("  Total      = {:.10} Hartree", mp2_result.total_energy);
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
) {
    let aux_name = cfg.mp2.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
    let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let mp3_result = mp3_energy(mol, prep, &dfbs, op, result, cfg.mp2.frozen_core)
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
) {
    let aux_name = cfg.mp2.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
    let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let oo_config = OoRiMp2Config {
        frozen_core: cfg.mp2.frozen_core,
        memory_budget_bytes: budget_bytes,
        verbose: cfg.scf.verbose,
        ..Default::default()
    };
    let oo_result = oo_ri_mp2(mol, prep, &dfbs, op, bounds, result, &oo_config)
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
    let aux_name = cfg.mp2.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
    let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let omega_ang_inv = cfg.mp2.omega.unwrap_or(0.420);
    let att_config = AttenuatedMp2Config {
        omega: omega_ang_inv * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV,
        scaling: 1.0,
        frozen_core: cfg.mp2.frozen_core,
        screen_thresh: None,
        memory_budget_bytes: budget_bytes,
    };
    let att_result = attenuated_ri_mp2(mol, prep, &dfbs, result, &att_config)
        .unwrap_or_else(|e| {
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
    println!("  E_OS       = {:.10} Hartree", att_result.spin_components.e_os);
    println!("  E_SS       = {:.10} Hartree", att_result.spin_components.e_ss);
    println!("  Total      = {:.10} Hartree", att_result.total_energy);
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
    let aux_name = cfg.mp2.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
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
            eprintln!("error: unknown [mp2] attenuator = \"{other}\"; expected \"erf\" or \"terf\"");
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
    if matches!(attenuator, ferric_rpa::rs_mp2_rpa::Attenuator::Terf)
        && cfg.mp2.omega.is_some()
    {
        eprintln!("warning: [mp2] omega is ignored when attenuator = \"terf\" (ω is derived from r0 = {r0_ang} Å = {r0:.4} Bohr as ω = 1/(r0·√2))");
    }

    // [mp2] r0_sweep: evaluate several r0 in one job, reusing the SCF above.
    // The SCF is already done by the time we get here, so each extra point
    // costs only the correlation stage — that is the whole point (an N-point
    // scan for ~1 SCF instead of N).
    let r0_sweep: Option<Vec<f64>> = cfg.mp2.r0_sweep.as_ref().map(|v| {
        let mut s: Vec<f64> = v.clone();
        s.sort_by(|a, b| a.partial_cmp(b).expect("r0_sweep must not contain NaN"));
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
            eprintln!("error: [mp2] r0_sweep requires attenuator = \"terf\" (r0 is meaningless for erf)");
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
        frozen_core: cfg.mp2.frozen_core,
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
            cfg, mol, bs, prep, &dfbs, aux_name, result, &rs_cfg, omega_ang_inv, r0_ang_k,
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
    let r = ferric_rpa::rs_mp2_rpa::rs_mp2_lr_rpa(mol, prep, dfbs, result, rs_cfg)
        .unwrap_or_else(|e| { eprintln!("error: {e}"); std::process::exit(1); });
    println!(
        "RS-MP2-RPA/{} (aux: {}, ω={:.3} Å⁻¹) on {}",
        bs.name, aux_name, omega_ang_inv, cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    match rs_cfg.attenuator {
        ferric_rpa::rs_mp2_rpa::Attenuator::Erf => {
            println!("RS-MP2-RPA [erf split] (ω = {omega_ang_inv:.3} Å⁻¹ = {:.4} Bohr⁻¹)", rs_cfg.omega);
        }
        ferric_rpa::rs_mp2_rpa::Attenuator::Terf => {
            let w_derived = 1.0 / (rs_cfg.r0 * std::f64::consts::SQRT_2);
            println!("RS-MP2-RPA [terf split] (r0 = {r0_ang:.4} Å = {:.4} Bohr, ω = 1/(r0·√2) = {:.4} Bohr⁻¹)", rs_cfg.r0, w_derived);
        }
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
    println!("  {:<20} = {:>16.10} Hartree", format!("E(SR-MP2, {sr_name})"), r.e_sr_mp2);
    println!("  {:<20} = {:>16.10} Hartree", format!("E(LR-MP2, {lr_name})"), r.e_lr_mp2);
    println!("  {:<20} = {:>16.10} Hartree", format!("E(dMP2, {lr_name})"), r.e_dmp2_lr);
    // Formulation-specific lines.
    match rs_cfg.formulation {
        ferric_rpa::RsMp2RpaFormulation::DeltaLr => {
            println!("  {:<20} = {:>16.10} Hartree", format!("E(dRPA, {lr_name})"), r.e_drpa_lr.unwrap());
            println!("  E_corr naive (A)     = {:>16.10} Hartree   [diagnostic: misses SR×LR cross terms]", r.e_corr_naive.unwrap());
            println!("  E_corr Δ-form (B)    = {:>16.10} Hartree", r.e_corr);
        }
        ferric_rpa::RsMp2RpaFormulation::CoupledRings => {
            println!("  E(ΔdRPA, Coulomb)    = {:>16.10} Hartree", r.e_delta_drpa_full.unwrap());
            println!("  {:<20} = {:>16.10} Hartree", format!("E(ΔdRPA, {sr_name})"), r.e_delta_drpa_sr.unwrap());
            println!("  E_corr coupled (T)   = {:>16.10} Hartree", r.e_corr);
        }
    }
    println!("  Total energy         = {:>16.10} Hartree", r.total_energy);
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
    let aux_name = cfg.mp2.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
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
        frozen_core: cfg.mp2.frozen_core,
        memory_budget_bytes: budget_bytes,
    };
    let scs_result = scs_mp2(mol, prep, &dfbs, result, &scs_config)
        .unwrap_or_else(|e| {
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
    let aux_name = cfg.mp2.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
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
        frozen_core: cfg.mp2.frozen_core,
        memory_budget_bytes: budget_bytes,
    };
    if scs_config.r0_nonbonded <= scs_config.r0_bonded {
        eprintln!("error: [mp2] r0_nonbonded must be > r0_bonded");
        std::process::exit(1);
    }
    let scs_result = scs_mp2_2terfc(mol, prep, &dfbs, result, &scs_config)
        .unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });
    println!(
        "SCS-MP2(2terfc)/{} (aux: {}, r0(1)={:.3} Å, r0(2)={:.3} Å, c_OS={:.3}, c_SS={:.3}) on {}",
        bs.name, aux_name, r0_bonded_ang, r0_nonbonded_ang, scs_config.c_os, scs_config.c_ss, cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!("  RHF energy = {:.10} Hartree", result.energy);
    println!("  SCS corr   = {:.10} Hartree", scs_result.scs_corr);
    println!("  E_OS       = {:.10} Hartree", scs_result.e_os);
    println!("  E_SS       = {:.10} Hartree", scs_result.e_ss);
    println!("  Total      = {:.10} Hartree", scs_result.total_energy);
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
    let aux_name = cfg.mp2.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
    let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let att_cfg = cfg.mp2.build_att_vv10_config(budget_bytes).unwrap_or_else(|e| {
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
    let ref_label = if is_closed_shell { "RHF energy " } else { "SCF energy " };
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
    println!(
        "MP2-V({attenuator})/{} (aux: {}, r0={:.3} Å, b={:.3}, C={:.4}, VV10 damping: {damping}) on {}",
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
    let aux_name = cfg.mp2.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
    let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let cc_config = CcConfig {
        frozen_core: cfg.mp2.frozen_core,
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
    let solver: &str = if is_closed_shell { "spin-adapted" } else { "spin-orbital" };
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
    println!("  CCSD corr  = {:.10} Hartree", cc_result.correlation_energy);
    println!("  Total      = {:.10} Hartree", result.energy + cc_result.correlation_energy);
}

/// `method.kind = "linlccd"`. Linearized hole-hole ladder CCD on the converged
/// closed-shell reference.
///
/// Mirrors [`run_ccsd`]'s aux-basis resolution (`[mp2] auxbasis`, default
/// `cc-pvdz-ri`) because LinLCCD is RI-based in exactly the same way. The
/// published [`LadderVariant::Hh`] is what is exposed: `DriversOnly` reproduces
/// RI-MP2 (already reachable via `method.kind = "rimp2"`) and `Full` carries
/// CCD-like VVVV memory, so neither earns a CLI knob here.
fn run_linlccd(
    cfg: &Config,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    op: Operator,
    result: &ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
) {
    let aux_name = cfg.mp2.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
    let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let cc_config = CcConfig {
        frozen_core: cfg.mp2.frozen_core,
        memory_budget_bytes: budget_bytes,
        ..Default::default()
    };
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
    let cc_result = linlccd(mol, prep, &dfbs, op, result, &cc_config, LadderVariant::Hh)
        .unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });
    println!(
        "LinLCCD(hh)/{} (aux: {}) on {}",
        bs.name, aux_name, cfg.molecule.xyz
    );
    println!("  nbasis     = {}", prep.nbasis());
    println!("  RHF energy = {:.10} Hartree", result.energy);
    println!("  LinLCCD corr = {:.10} Hartree", cc_result.correlation_energy);
    println!("  Total      = {:.10} Hartree", result.energy + cc_result.correlation_energy);
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
    let aux_name = cfg.mp2.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
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
            frozen_core: cfg.mp2.frozen_core,
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

/// `method.kind = "laplace-mp2"`. Extracted verbatim from the former
/// `main()` `"laplace-mp2" => { ... }` match arm.
fn run_laplace_mp2(
    cfg: &Config,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    op: Operator,
    result: &ferric_scf::result::ScfResult,
) {
    let aux_name = cfg.mp2.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
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
        cfg.mp2.frozen_core,
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
    let aux_name = cfg.mp2.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
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
        frozen_core: cfg.mp2.frozen_core,
        n_quad: cfg.mp2.n_quad.unwrap_or(7),
        memory_budget_bytes: budget_bytes,
        domain_cutoff_bohr: cfg.mp2.domain_cutoff_bohr,
    };
    let sos = laplace_sos_mp2(mol, prep, &dfbs, op, result, &sos_cfg, formulation)
        .unwrap_or_else(|e| {
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
    proatom_gs_mult: &dyn Fn(i32) -> usize,
    proatom: &dyn Fn(i32, i32) -> Option<ferric_rpa::properties::RadialProatom>,
) {
        let aux_name = cfg.rpa.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
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
            frozen_core: cfg.rpa.frozen_core,
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
            let uhf_result = solve_uhf(ctx, mol, prep, bounds, &uhf_cfg)
                .unwrap_or_else(|e| {
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
            let rr = run_pdep_rpa(mol, prep, &dfbs, op, &result, &rpa_cfg)
                .unwrap_or_else(|e| {
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
        println!("{ref_label} energy:            {:>20.10} Hartree", result.energy);
        println!("RPA correlation:       {:>20.10} Hartree", rpa_result.e_rpa);
        println!("Total ({ref_label}+RPA):       {:>20.10} Hartree", result.energy + rpa_result.e_rpa);
        println!("Eigenpotentials kept:  {} / {}", rpa_result.n_eigenpotentials, rpa_result.eigenvalues_static.len());
        if let Some(e_diag) = rpa_result.e_rpa_dft_diag {
            println!("RI-dRPA check:         {:>20.10} Hartree", e_diag);
        }
        if let Some(prefix) = cfg.rpa.export_eigpot_prefix.as_deref() {
            use ferric_export::cube::GridSpec;
            use ferric_export::export_basis_function_cube;
            let spacing = cfg.rpa.cube_spacing.unwrap_or(0.2);
            let margin = cfg.rpa.cube_margin.unwrap_or(4.0);
            let n_export = cfg.rpa.export_eigpot_count
                .unwrap_or(10)
                .min(rpa_result.n_eigenpotentials);
            let grid = GridSpec::bounding_box(mol, margin, spacing);
            println!(
                "Exporting {} eigenpotential cubes (grid {}×{}×{}, spacing {} Bohr)…",
                n_export, grid.n_x, grid.n_y, grid.n_z, spacing
            );
            for alpha in 0..n_export {
                let coeffs: Vec<f64> = rpa_result.eigenpotentials
                    .column(alpha).iter().copied().collect();
                let lam = rpa_result.eigenvalues_static[alpha];
                let path = format!("{prefix}_eigpot_{:03}.cube", alpha);
                let comment = format!(
                    "PDEP eigenpotential α={alpha} λ(0)={lam:.6} (basis {aux_name})"
                );
                if let Err(e) = export_basis_function_cube(&path, mol, &aux_bs, &grid, &coeffs, &comment) {
                    eprintln!("  warning: failed to write {}: {}", path, e);
                } else {
                    println!("  wrote {} (λ(0)={:.6})", path, lam);
                }
            }
        }
        // NPZ feature bundle for diffusion-model export.
        if let Some(npz_path) = cfg.rpa.export_npz.as_deref() {
            use ferric_export::export_npz;
            use ferric_export::ml::{ChargeSchemes, DispersionBundle, NpzBundle, PolarizabilityBundle};
            use ferric_rpa::properties::{
                chelpg_and_resp_charges, chelpg_charges, electric_field_at_atoms, esp_at_atoms,
                hirshfeld_charges,
                lowdin_charges, mulliken_charges,
                pdep_polarizability_becke,
                pdep_polarizability_static,
                resp_charges,
            };
            use ndarray::Array2;

            let compute_esp = cfg.rpa.compute_esp.unwrap_or(true);
            let compute_pol = cfg.rpa.compute_polarizability.unwrap_or(true);
            let compute_ef = cfg.rpa.compute_electric_field.unwrap_or(true);
            let compute_alpha_atomic = cfg.rpa.compute_alpha_atomic.unwrap_or(true);

            let coords_arr = {
                let mut a = Array2::<f64>::zeros((mol.atoms.len(), 3));
                for (i, atom) in mol.atoms.iter().enumerate() {
                    a[(i, 0)] = atom.x;
                    a[(i, 1)] = atom.y;
                    a[(i, 2)] = atom.zpos;
                }
                a
            };
            let znums: Vec<usize> =
                mol.atoms.iter().map(|a| a.z as usize).collect();

            let esp_vec = if compute_esp {
                match esp_at_atoms(mol, prep, result.density_total()) {
                    Ok(v) => Some(v),
                    Err(e) => {
                        eprintln!("warning: esp_at_atoms failed: {e}");
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
                        None
                    }
                }
            } else {
                None
            };

            let alpha_arr = if compute_pol {
                match pdep_polarizability_static(
                    mol, prep, &dfbs, &result, op, &rpa_cfg,
                ) {
                    Ok(p) => {
                        println!(
                            "Polarizability α (a.u.):  iso={:.4}, principal=[{:.4}, {:.4}, {:.4}]",
                            p.iso, p.principal[0], p.principal[1], p.principal[2]
                        );
                        Some(p.tensor)
                    }
                    Err(e) => {
                        eprintln!("warning: polarizability failed: {e}");
                        None
                    }
                }
            } else {
                None
            };

            let alpha_atomic_vec = if compute_alpha_atomic {
                match pdep_polarizability_becke(
                    mol, prep, bs, &dfbs, &result, op, &rpa_cfg,
                ) {
                    Ok(v) => {
                        println!(
                            "Per-atom Becke α (iso, a.u.): {:?}",
                            v.iter()
                                .map(|t| (t[0][0] + t[1][1] + t[2][2]) / 3.0)
                                .collect::<Vec<_>>()
                        );
                        Some(v)
                    }
                    Err(e) => {
                        eprintln!("warning: per-atom α (Hirshfeld) failed: {e}");
                        None
                    }
                }
            } else {
                None
            };

            let compute_dm = cfg.rpa.compute_density_matrix.unwrap_or(true);
            let dm_ref = if compute_dm { Some(result.density_total()) } else { None };

            // Molecular dipole μ = −Tr(P·D) + Σ_A Z_A R_A of the total density
            // (QC ground truth vs partition-derived Löwdin/Hirshfeld dipoles).
            // Origin [0,0,0]; neutral molecules → origin-independent. Mirrors
            // ferric-mp2 ff_polar::mp2_dipole; P·D summed elementwise = Tr(P·D)
            // since both AO matrices are symmetric.
            let compute_dip = cfg.rpa.compute_dipole.unwrap_or(true);
            let dip_arr: Option<[f64; 3]> = if compute_dip {
                match ferric_integrals::oneelectron::dipole(prep, [0.0, 0.0, 0.0]) {
                    Ok(dip_ao) => {
                        let p = result.density_total();
                        let mut mu = [0.0f64; 3];
                        for d in 0..3 {
                            let elec = (p * &dip_ao[d]).sum();
                            let nuc: f64 = mol
                                .atoms
                                .iter()
                                .map(|a| a.z as f64 * [a.x, a.y, a.zpos][d])
                                .sum();
                            mu[d] = nuc - elec;
                        }
                        println!(
                            "dipole (e·a0): [{:.4}, {:.4}, {:.4}] |μ| = {:.4}",
                            mu[0], mu[1], mu[2], (mu[0] * mu[0] + mu[1] * mu[1] + mu[2] * mu[2]).sqrt()
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
                            q.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>()
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
                            q.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>()
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
                            q.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>()
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
            fn fmt_q(q: &[f64]) -> Vec<f64> { q.iter().map(|v| (v * 1e4).round() / 1e4).collect() }
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
            let mut c6_iso_opt: Option<ndarray::Array2<f64>> = None;
            let mut c6_aniso_v: Vec<Vec<[[f64; 3]; 3]>> = Vec::new();
            if compute_c6 {
                use ferric_rpa::dispersion::{
                    casimir_polder_c6, pdep_dynamic_polarizability,
                    ts_dynamic_polarizability, C6Source, DispersionPartition,
                };
                use ferric_rpa::properties::{
                    atomic_effective_volumes_hirshfeld,
                    pdep_polarizability_hirshfeld,
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
                let partition =
                    DispersionPartition::parse_config_str(cfg.rpa.c6_partition.as_deref())
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
                        mol, prep, bs, &dfbs, &result, op, &rpa_cfg, partition,
                        Some(proatom),
                    ) {
                        Ok(dp) => {
                            let res = casimir_polder_c6(&dp);
                            println!(
                                "Computed PDEP-RPA C6: {} atoms, {} freqs; molecular C6 = {:.3} a.u.",
                                mol.atoms.len(), dp.freqs.len(), res.c6_molecular_iso
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
                            mol, prep, bs, &dfbs, &result, op, &rpa_cfg, Some(proatom),
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
                        mol, bs, result.density_total(), Some(proatom),
                    ) {
                        Ok(v) => v,
                        Err(e) => {
                            eprintln!("warning: TS C6 skipped — Hirshfeld effective volumes failed: {e}");
                            return None;
                        }
                    };
                    let z: Vec<usize> = mol.atoms.iter().map(|a| a.z as usize).collect();

                    // Compute free-atom vol_free using Hirshfeld on isolated atoms.
                    // For a single atom Hirshfeld weight = 1 everywhere (only one
                    // proatom), so this gives ∫ ρ_free(r) |r|³ dr — same physics
                    // as the molecular Hirshfeld integral, consistent denominator.
                    let mut vol_free_computed: std::collections::HashMap<usize, f64> =
                        std::collections::HashMap::new();
                    for &zi in z.iter().collect::<std::collections::HashSet<_>>() {
                        let sym = ferric_core::elements::z_to_symbol(zi as i32)
                            .unwrap_or("X");
                        let free_xyz = format!("1\n{sym}\n{sym} 0 0 0\n");
                        // Correct atomic ground-state multiplicities (3P for
                        // C/O/Si/S, etc.). Reuse the proatom map — the prior
                        // ad-hoc match here gave C/O/S a singlet, which is
                        // wrong physics and HANGS the restricted SCF for S.
                        let mult = proatom_gs_mult(zi as i32);
                        if let Ok(free_mol) = Molecule::parse_xyz(&free_xyz, 0, mult) {
                            if let Ok(free_obs) = PreparedBasis::new(&free_mol, bs) {
                                let free_bounds = SchwarzBounds::compute(op, &free_obs)
                                    .unwrap_or_else(|_| SchwarzBounds::compute(op, prep).unwrap());
                                let mut free_cfg = rhf_config.clone();
                                free_cfg.mom_after_iter = if mult > 1 { 5 } else { 0 };
                                // Give the tiny free-atom SCF a generous iteration
                                // budget — this is now the ONLY source of vol_free
                                // (the hardcoded-table fallback was removed), so a
                                // near-converged atom that would previously have
                                // silently degraded to a table value must instead
                                // actually converge. Cheap: it's a single atom.
                                free_cfg.max_iter = free_cfg.max_iter.max(200);
                                // 1-thread pool for the tiny atom solve — see run_serial.
                                //
                                // The free-atom volume must be on the SAME scale (same xc) as
                                // the molecular volume (vols[i]) or the ratio is meaningless.
                                // Open-shell xc atoms (³P: O/S/Si) do NOT converge under a
                                // plain UKS-GGA solve — their degenerate p-shell makes the GGA
                                // potential orientation-dependent and the SCF oscillates
                                // forever. Fractional/ensemble occupation (fractional_occ)
                                // spreads the open-shell electrons equally over the degenerate
                                // p orbitals, restoring spherical symmetry and converging the
                                // UKS-PBE atom on the *consistent* scale. Pure HF/UHF free-atom
                                // solves don't suffer this (K is orbital-invariant in the
                                // degenerate subspace), so — matching the proatom builder above
                                // — only enable fractional_occ when an xc functional is set.
                                if mult > 1 && free_cfg.xc.is_some() {
                                    free_cfg.fractional_occ = true;
                                }
                                let solve_free = |cfg: &RhfConfig| -> Option<ndarray::Array2<f64>> {
                                    if mult > 1 {
                                        solve_uhf(ctx, &free_mol, &free_obs, &free_bounds, cfg)
                                            .ok().map(|r| r.density_total().to_owned())
                                    } else {
                                        solve_rhf(ctx, &free_mol, &free_obs, op, &free_bounds, cfg)
                                            .ok().map(|r| r.density_r().to_owned())
                                    }
                                };
                                // Live free-atom SCF is the ONLY source of the TS
                                // vol_free denominator now. Try the reference-
                                // consistent xc solve first (scale-matched to the
                                // molecular volume); if it fails, retry pure HF/UHF
                                // as a *scale-consistent* fallback (this changes the
                                // xc convention slightly, but is still a real
                                // free-atom integral, not a stale table number). If
                                // both fail, vol_free_computed has no entry for this
                                // Z and the loop below skips TS C6 with a clear
                                // warning — no silent scale-mismatched fabrication.
                                let free_density = run_serial(|| {
                                    solve_free(&free_cfg).or_else(|| {
                                        // xc solve failed — retry pure HF/UHF for a converged,
                                        // scale-consistent density.
                                        let mut hf_cfg = free_cfg.clone();
                                        hf_cfg.xc = None;
                                        hf_cfg.fractional_occ = false;
                                        solve_free(&hf_cfg)
                                    })
                                });
                                if let Some(d) = free_density {
                                    // Single free atom: Hirshfeld weight = 1
                                    // everywhere (one proatom), so the
                                    // reference volume is partition-independent
                                    // — None (legacy path) is exact here.
                                    if let Ok(fv) = atomic_effective_volumes_hirshfeld(
                                        &free_mol, bs, &d, None,
                                    ) {
                                        vol_free_computed.insert(zi, fv[0]);
                                    }
                                }
                            }
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
                                     sourced for most elements (docs/vol-free-verification.md). \
                                     Refusing to fabricate a C6 from a mismatched denominator \
                                     (same convention as the Z>18 hard-error path — see \
                                     ts_atom_params / CLAUDE.md TS/MBD honesty). Use \
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
                            &positions, &z, &ratio, &alpha_static, &freqs, &weights,
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
                    c6_freqs_v = res.per_atom_dynamic.freqs.clone();
                    c6_weights_v = res.per_atom_dynamic.weights.clone();
                    alpha_dyn_v = res.per_atom_dynamic.per_atom.clone();
                    c6_iso_opt = Some(res.c6_iso_pair.clone());
                    c6_aniso_v = res.c6_aniso_pair.clone();
                }
            }

            let npz_bundle = NpzBundle {
                mo_coeffs: if result.spin == ferric_scf::result::Spin::Restricted { Some(result.mos_r()) } else { None },
                orbital_energies: if result.spin == ferric_scf::result::Spin::Restricted { Some(result.eps_r()) } else { None },
                pdep_eigenvectors: Some(&rpa_result.eigenpotentials),
                boys_coeffs: None,
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
                    alpha_tensor: alpha_arr.as_ref(),
                    electric_field: ef_vec.as_deref(),
                    alpha_atomic: alpha_atomic_vec.as_deref(),
                },
                dispersion: DispersionBundle {
                    c6_freqs: if c6_freqs_v.is_empty() { None } else { Some(c6_freqs_v.as_slice()) },
                    c6_weights: if c6_weights_v.is_empty() { None } else { Some(c6_weights_v.as_slice()) },
                    alpha_atomic_dynamic: if alpha_dyn_v.is_empty() { None } else { Some(alpha_dyn_v.as_slice()) },
                    c6_iso: c6_iso_opt.as_ref(),
                    c6_aniso: if c6_aniso_v.is_empty() { None } else { Some(c6_aniso_v.as_slice()) },
                },
            };
            if let Err(e) = export_npz(npz_path, &npz_bundle) {
                eprintln!("warning: failed to write {}: {}", npz_path, e);
            } else {
                println!("Wrote NPZ feature bundle: {}", npz_path);
                if c6_iso_opt.is_some() {
                    println!(
                        "note: NPZ c6_iso/c6_aniso are per-atom PAIR tensors, not the \
                         molecular C6 total — do not sum them to approximate it (can be \
                         20-58% off; see the \"molecular C6 = ... a.u.\" line above for the \
                         correct DOSD-comparable value, or docs/dosd-c6-rpa-vs-ts.md)."
                    );
                }
            }
        }
}

/// `method.kind = "gw"`. Extracted verbatim from the former `main()`
/// `"gw" => { ... }` match arm.
#[allow(clippy::too_many_arguments)]
fn run_gw(
    cfg: &Config,
    ctx: &ParallelContext,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf_config: &RhfConfig,
    result: &ferric_scf::result::ScfResult,
    budget_bytes: Option<usize>,
) {
        let aux_name = cfg.rpa.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
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
        let gw_frozen_core = cfg.gw.frozen_core.unwrap_or(cfg.rpa.frozen_core);
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
            // Open-shell path: re-run with UHF + MOM (same precedent as the
            // "pdep-rpa" arm's open-shell dispatch) so the reference is
            // converged, then dispatch to run_u_gw. Shadow `result` so it
            // carries the correct (possibly UKS) SCF density.
            let mut uhf_cfg = rhf_config.clone();
            uhf_cfg.mom_after_iter = 5;
            let result = solve_uhf(ctx, mol, prep, bounds, &uhf_cfg).unwrap_or_else(|e| {
                eprintln!("error (UHF): {e}");
                std::process::exit(1);
            });
            // KS reference (RPA@PBE0-style): [rpa].xc set ⇒ `result` above is
            // already the UKS solve; build vxc_diag_a/b and apply the Σx−vxc
            // shift post-hoc via UGwResult::apply_kohn_sham_correction (U-GW
            // doesn't thread vxc_diag through run_u_gw itself — see its doc).
            // None (HF reference) ⇒ no shift, matches run_u_gw's contract.
            let vxc_diag = match cfg.rpa.xc.as_deref() {
                Some(xc_name) => {
                    let (diag_a, diag_b) =
                        ferric_gw::vxc_mo::vxc_diagonal_mo(mol, bs, xc_name, &result)
                            .unwrap_or_else(|e| {
                                eprintln!("error: vxc_diagonal_mo failed: {e}");
                                std::process::exit(1);
                            });
                    Some((diag_a, diag_b))
                }
                None => None,
            };
            let mut gw_result = ferric_gw::run_u_gw(
                mol, prep, &dfbs, op, &result, &rpa_cfg, &gw_cfg,
            )
            .unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            if let Some((diag_a, diag_b)) = vxc_diag.as_ref() {
                gw_result.apply_kohn_sham_correction(diag_a, diag_b);
            }
            let ref_label = if cfg.rpa.xc.is_some() { "UKS" } else { "UHF" };
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
                    "alpha", nocc_a,
                    &gw_result.eps_mf_a, &gw_result.eps_qp_a, &gw_result.sigma_x_a,
                    &gw_result.sigma_c_a, &gw_result.z_factor_a, &gw_result.qp_converged_a,
                ),
                (
                    "beta", nocc_b,
                    &gw_result.eps_mf_b, &gw_result.eps_qp_b, &gw_result.sigma_x_b,
                    &gw_result.sigma_c_b, &gw_result.z_factor_b, &gw_result.qp_converged_b,
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
            mol, prep, &dfbs, op, result, &rpa_cfg, &gw_cfg, vxc_diag.as_ref(),
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
        let aux_name = cfg.rpa.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
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
        let bse_frozen_core = cfg.gw.frozen_core.unwrap_or(cfg.rpa.frozen_core);
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
        let bse = ferric_gw::bse::run_bse_tda(
            mol, prep, &dfbs, op, result, &rpa_cfg, bse_frozen_core,
        )
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
        println!("  nocc = {}  nvir = {}  ({} singlet states)", bse.nocc, bse.nvir, bse.omega.len());
        println!("  {:>4} {:>12} {:>10}", "n", "Omega (eV)", "f_osc");
        for (n, (&om, &f)) in bse.omega.iter().zip(bse.oscillator_strength.iter()).enumerate() {
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
        // This method's validated accuracy (static alpha ~= DOSD) is a
        // KS-reference result; require [rpa].xc explicitly rather than
        // silently falling back to an HF reference with a much worse
        // static alpha (see the xc-routing block's comment above).
        if cfg.rpa.xc.is_none() {
            eprintln!(
                "error: method.kind = \"tdhf-static-polarizability\" requires [rpa] xc \
                 (e.g. xc = \"PBE\") -- this method's validated accuracy is a KS-reference \
                 result; an HF reference gives a much worse static alpha"
            );
            std::process::exit(1);
        }
        let aux_name = cfg.rpa.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
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
        let frozen_core = cfg.gw.frozen_core.unwrap_or(cfg.rpa.frozen_core);
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
            mol, prep, &dfbs, op, result, &rpa_cfg, frozen_core, scissor,
        )
        .unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });
        println!(
            "RPAx@KS[{}] static polarizability /{} (aux: {}) on {}",
            cfg.rpa.xc.as_deref().unwrap_or("?"), bs.name, aux_name, cfg.molecule.xyz
        );
        println!(
            "  NOTE: static polarizability only -- do not use for C6/dispersion \
             (known negative accuracy result, see docs/VALIDATION.md)"
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

/// `method.kind = "uhf"`, `task = "energy"`. Extracted verbatim from the
/// former `main()` `if method == "uhf" { ... }` block.
#[allow(clippy::too_many_arguments)]
fn run_uhf(
    cfg: &Config,
    ctx: &ParallelContext,
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    rhf_config: &RhfConfig,
) {
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
    println!("UHF/{} on {}", bs.name, cfg.molecule.xyz);
    println!("  nbasis     = {}", prep.nbasis());
    println!("  mult       = {} (nocc_a={}, nocc_b={})", mol.multiplicity, nocc_a, nocc_b);
    println!("  iterations = {}", result.iterations);
    println!("  converged  = {}", result.converged);
    println!("  energy     = {:.10} Hartree", result.energy);
    println!("  <S^2>      = {:.6} (ideal {:.6})", s2, s_ideal);
    // task == "optimize" is handled by the top-level dispatch above
    // (optimize_geometry_uhf), which returns before reaching here.
}

/// `method.kind = "rohf"`, `task = "energy"`. Extracted verbatim from the
/// former `main()` `if method == "rohf" { ... }` block.
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
    println!("ROHF/{} on {}", bs.name, cfg.molecule.xyz);
    println!("  nbasis     = {}", prep.nbasis());
    println!(
        "  mult       = {} (nocc_double={}, nocc_open={})",
        mol.multiplicity, nocc_double, nocc_open
    );
    println!("  iterations = {}", result.iterations);
    println!("  converged  = {}", result.converged);
    println!("  energy     = {:.10} Hartree", result.energy);
    println!("  <S^2>      = {:.6} (exact by construction)", s_ideal);
    // task == "optimize" is handled by the top-level dispatch above
    // (optimize_geometry_rohf), which returns before reaching here.
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
    };
    match method {
        "rhf" | "ksdft" => {
            let opt_result = optimize_geometry(ctx, mol, &bs.name, op, rhf_config, &opt_config)
                .unwrap_or_else(|e| {
                    eprintln!("error during optimization: {e}");
                    std::process::exit(1);
                });
            println!("\nFinal Optimized Geometry (Bohr):");
            for (i, atom) in opt_result.mol.atoms.iter().enumerate() {
                println!("  {:2} {:2} {:12.8} {:12.8} {:12.8}", i, atom.symbol, atom.x, atom.y, atom.zpos);
            }
            println!("\nOptimization Result:");
            println!("  converged  = {}", opt_result.converged);
            println!("  steps      = {}", opt_result.steps);
            println!("  final E    = {:.10} Hartree", opt_result.energy);
        }
        "pdep-rpa" => {
            let aux_name = cfg.rpa.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
            let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            let scheme = cfg.rpa.parse_quadrature().unwrap_or_else(|e| {
                eprintln!("config error: {e}");
                std::process::exit(1);
            });
            let rpa_cfg = PdepRpaConfig {
                frozen_core: cfg.rpa.frozen_core,
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
            let opt_result =
                ferric_rpa::optimize::optimize_geometry_rpa(mol, bs, &aux_bs, op, &rpa_cfg, &opt_config, h_fd)
                    .unwrap_or_else(|e| {
                        eprintln!("error during RPA optimization: {e}");
                        std::process::exit(1);
                    });
            println!("\nFinal Optimized Geometry (Bohr):");
            for (i, atom) in opt_result.mol.atoms.iter().enumerate() {
                println!("  {:2} {:2} {:12.8} {:12.8} {:12.8}", i, atom.symbol, atom.x, atom.y, atom.zpos);
            }
            println!("\nRPA Optimization Result:");
            println!("  converged  = {}", opt_result.converged);
            println!("  steps      = {}", opt_result.steps);
            println!("  final E    = {:.10} Hartree (RHF + RPA)", opt_result.energy);
        }
        "rimp2" => {
            let aux_name = cfg.mp2.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
            let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            let mp2_config = RiMp2Config {
                frozen_core: cfg.mp2.frozen_core,
                memory_budget_bytes: budget_bytes,
                ..Default::default()
            };
            let opt_result = ferric_mp2::optimize::optimize_geometry_rimp2(
                mol, bs, &aux_bs, op, &mp2_config, &opt_config,
            )
            .unwrap_or_else(|e| {
                eprintln!("error during RI-MP2 optimization: {e}");
                std::process::exit(1);
            });
            println!("\nFinal Optimized Geometry (Bohr):");
            for (i, atom) in opt_result.mol.atoms.iter().enumerate() {
                println!("  {:2} {:2} {:12.8} {:12.8} {:12.8}", i, atom.symbol, atom.x, atom.y, atom.zpos);
            }
            println!("\nRI-MP2 Optimization Result:");
            println!("  converged  = {}", opt_result.converged);
            println!("  steps      = {}", opt_result.steps);
            println!("  final E    = {:.10} Hartree (RHF + MP2)", opt_result.energy);
        }
        "uhf" => {
            let opt_result = optimize_geometry_uhf(ctx, mol, &bs.name, op, rhf_config, &opt_config)
                .unwrap_or_else(|e| {
                    eprintln!("error during UHF optimization: {e}");
                    std::process::exit(1);
                });
            println!("\nFinal Optimized Geometry (Bohr):");
            for (i, atom) in opt_result.mol.atoms.iter().enumerate() {
                println!("  {:2} {:2} {:12.8} {:12.8} {:12.8}", i, atom.symbol, atom.x, atom.y, atom.zpos);
            }
            println!("\nUHF Optimization Result:");
            println!("  converged  = {}", opt_result.converged);
            println!("  steps      = {}", opt_result.steps);
            println!("  final E    = {:.10} Hartree", opt_result.energy);
        }
        "rohf" => {
            let opt_result = optimize_geometry_rohf(ctx, mol, &bs.name, op, rhf_config, &opt_config)
                .unwrap_or_else(|e| {
                    eprintln!("error during ROHF optimization: {e}");
                    std::process::exit(1);
                });
            println!("\nFinal Optimized Geometry (Bohr):");
            for (i, atom) in opt_result.mol.atoms.iter().enumerate() {
                println!("  {:2} {:2} {:12.8} {:12.8} {:12.8}", i, atom.symbol, atom.x, atom.y, atom.zpos);
            }
            println!("\nROHF Optimization Result:");
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
