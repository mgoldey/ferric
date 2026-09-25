//! `[cell]`: periodic runs through the `ferric-pbc` drivers.
//!
//! `run()` hands the whole run here when the config carries a validated
//! [`PeriodicPlan`] (resolved and checked by `config::periodic_plan` at load
//! time, so every refusal -- a method with no periodic driver, a k-mesh on a
//! Gamma-only route, a knob the route would ignore -- has already happened).
//! This module only assembles and calls the drivers, exactly as the Python
//! periodic bindings do (`crates/ferric-python/src/pbc.rs` and
//! `run_rhf_gamma`): the same hcore -> dense-AFT / RS-GDF J/K -> injected SCF
//! assembly, the same configs, the same defaults. No new physics.
//!
//! Units: the XYZ is Ångström (as always), the lattice was converted to Bohr
//! by the plan; energies are Hartree PER CELL.

use crate::config::{Config, PeriodicJk, PeriodicPlan, PeriodicRoute, ANGSTROM_TO_BOHR};
use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_pbc::dense_aft::DEFAULT_DENSE_AFT_PRECISION;
use ferric_pbc::drpa::DEFAULT_GAMMA_DRPA_QUAD_POINTS;
use ferric_pbc::kcorr::DEFAULT_KDRPA_QUAD_POINTS;
use ferric_pbc::{
    gamma_drpa, gamma_mp2, gamma_rhf_gradient, gamma_rks, gamma_rohf, gamma_roks, gamma_uhf,
    gamma_uks, kpoint_drpa, kpoint_mp2, periodic_hcore, periodic_hcore_kpts, solve_krhf,
    solve_krhf_injected, solve_kuhf, Cell, DenseAftEri, ExxDiv, GammaDrpaConfig,
    GammaDrpaIntegrals, GammaMp2Config, GammaMp2Integrals, GammaRksConfig, GammaRohfConfig,
    GammaRoksConfig, GammaUhfConfig, GammaUhfIntegrals, GammaUksConfig, KCorrIntegrals,
    KDenseAftConfig, KDenseAftEri, KDenseAftPairs, KDrpaConfig, KDrpaEnergy, KJkKind, KMp2Config,
    KPointInjection, KPointMesh, KRhfConfig, KRsGdf, KRsGdfConfig, KScfConfig, KScfResult,
    KUhfConfig, MeshCentring, Mp2Denominators, PeriodicGridConfig, PeriodicHcore,
    PeriodicHcoreConfig, RsGdf, RsGdfConfig,
};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;

/// Print `error: {msg}` and exit 1, the CLI's convention.
fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1)
}

/// What every periodic handler reports back for the run log.
struct Outcome {
    energy: f64,
    converged: bool,
    /// `"total"` for SCF routes, `"correlated_total"` for MP2/dRPA.
    energy_is: &'static str,
}

/// Entry point from `run()`: `cfg.periodic` must be `Some`.
/// The SCF iteration cap a route actually runs with: `[scf] max_iter` when
/// written; otherwise the ROKS route takes `GammaRoksConfig::new`'s default
/// (600 for hybrids) and every other route the plan's.
fn effective_max_iter(plan: &PeriodicPlan) -> usize {
    match (plan.route, plan.max_iter_explicit) {
        (PeriodicRoute::Roks, false) => {
            GammaRoksConfig::new(plan.functional.as_deref().unwrap_or(""))
                .scf
                .max_iter
        }
        _ => plan.max_iter,
    }
}

pub fn run_periodic(cfg: &Config) {
    let Some(plan) = cfg.periodic.as_ref() else {
        die("internal: run_periodic called without a periodic plan");
    };
    let s = setup(cfg, plan);
    if let Some(rl) = ferric_scf::runlog::log() {
        rl.run_start(
            serde_json::json!({
                "method": cfg.method.kind,
                "task": cfg.method.task,
                "basis": s.bs.name,
                "periodic_route": plan.route.label(),
                "functional": plan.functional,
                "lattice_bohr": plan.lattice_bohr,
                "kmesh": plan.kmesh.map(|(n, _)| n),
                "exxdiv": exx_name(plan.exxdiv),
                "jk": jk_name(plan),
                "max_iter": effective_max_iter(plan),
            }),
            serde_json::json!({
                "path": cfg.molecule.xyz,
                "n_atoms": s.cell.mol().atoms.len(),
                "charge": cfg.molecule.charge,
                "multiplicity": cfg.molecule.multiplicity,
                "n_electrons": s.cell.mol().nelec(),
                "n_basis": s.prep.nbasis(),
            }),
        );
    }
    let out = if plan.optimize {
        run_gamma_rhf_optimize(cfg, plan, &s)
    } else if plan.kmesh.is_some() {
        match plan.route {
            PeriodicRoute::Rhf => run_krhf(cfg, plan, &s),
            PeriodicRoute::Uhf => run_kuhf(cfg, plan, &s),
            PeriodicRoute::Mp2 | PeriodicRoute::Drpa => run_kcorr(cfg, plan, &s),
            // Refused by `periodic_plan` (`PeriodicRoute::has_kpoints`).
            other => die(format!(
                "internal: k-point {} reached the periodic dispatcher",
                other.label()
            )),
        }
    } else {
        match plan.route {
            PeriodicRoute::Rhf => run_gamma_rhf(cfg, plan, &s),
            PeriodicRoute::Rks => run_gamma_rks(cfg, plan, &s),
            PeriodicRoute::Uhf | PeriodicRoute::Uks | PeriodicRoute::Rohf | PeriodicRoute::Roks => {
                run_gamma_open(cfg, plan, &s)
            }
            PeriodicRoute::Mp2 | PeriodicRoute::Drpa => run_gamma_corr(cfg, plan, &s),
        }
    };
    if let Some(rl) = ferric_scf::runlog::log() {
        rl.run_end(
            out.energy,
            out.converged,
            if out.converged {
                "converged"
            } else {
                "not_converged"
            },
            serde_json::json!({
                "method": cfg.method.kind,
                "task": cfg.method.task,
                "periodic": true,
                "energy_is": out.energy_is,
                "energy_unit": "hartree_per_cell",
            }),
        );
    }
}

// ───────────────────────────────────────────────────────────────── setup ──

/// A validated cell, orbital basis and (RS-GDF) aux basis.
struct Setup {
    bs: BasisSet,
    cell: Cell,
    prep: PreparedBasis,
    /// RS-GDF aux basis on the cell's atoms; `None` = dense AFT.
    aux: Option<PreparedBasis>,
    /// Nuclear-attraction Ewald split (Bohr⁻¹), resolved.
    omega_bohr: f64,
}

fn load_basis(cfg: &Config) -> BasisSet {
    if let Some(name) = &cfg.basis.name {
        basis::bundled(name)
    } else if let Some(path) = &cfg.basis.path {
        basis::load_g94(path)
    } else {
        Err(FerricError::Basis("no basis specified".into()))
    }
    .unwrap_or_else(|e| die(e))
}

fn setup(cfg: &Config, plan: &PeriodicPlan) -> Setup {
    let mut mol = Molecule::load_xyz_with_charge(
        &cfg.molecule.xyz,
        cfg.molecule.charge,
        cfg.molecule.multiplicity,
    )
    .unwrap_or_else(|e| die(e));
    let bs = load_basis(cfg);
    // The ECP (if the basis carries one) is applied before the Cell is built;
    // periodic_hcore(_kpts) adds the lattice-summed V_ECP and re-checks it.
    mol.apply_ecp(&bs);
    if plan.route.closed_shell() {
        let nelec = mol.nelec();
        if nelec <= 0 || nelec % 2 != 0 {
            die(format!(
                "[cell]: {nelec} electrons per cell; the periodic {} driver is closed-shell \
                 and needs a positive even count (use method.kind = \"uhf\"/\"rohf\")",
                plan.route.label()
            ));
        }
    }
    let cell = Cell::new(mol, plan.lattice_bohr).unwrap_or_else(|e| die(format!("[cell]: {e}")));
    let prep = PreparedBasis::new(cell.mol(), &bs).unwrap_or_else(|e| die(e));
    let aux = match &plan.jk {
        PeriodicJk::Dense { max_eri_bytes } => {
            // The Gamma dense tensor is refused before any lattice sum (the
            // Rust builder re-checks the same bound). The k-point builders
            // size and check their own kernels.
            if plan.kmesh.is_none() {
                let nao = prep.nbasis();
                let need = 8u128 * (nao as u128).pow(4);
                if need > *max_eri_bytes as u128 {
                    die(format!(
                        "[cell]: the dense AFT ERI tensor needs {need} bytes (nao = {nao}) > \
                         max_eri_gb cap {max_eri_bytes} bytes. This path is a toy-scale \
                         oracle; use jk = \"rsgdf\" with an auxbasis for larger cells."
                    ));
                }
            }
            None
        }
        PeriodicJk::RsGdf { auxbasis, .. } => {
            let a = basis::bundled(auxbasis).unwrap_or_else(|e| {
                die(format!(
                    "[cell] auxbasis {auxbasis:?} is not a bundled basis: {e}"
                ))
            });
            Some(PreparedBasis::new(cell.mol(), &a).unwrap_or_else(|e| {
                die(format!(
                    "[cell] auxbasis {auxbasis:?} cannot be placed on this cell: {e}"
                ))
            }))
        }
    };
    let omega_bohr = plan
        .omega_bohr
        .unwrap_or_else(|| ferric_pbc::ewald::default_ewald_omega(&cell));
    Setup {
        bs,
        cell,
        prep,
        aux,
        omega_bohr,
    }
}

fn exx_name(exx: ExxDiv) -> &'static str {
    match exx {
        ExxDiv::Ewald => "ewald",
        ExxDiv::None => "none",
    }
}

fn jk_name(plan: &PeriodicPlan) -> String {
    match &plan.jk {
        PeriodicJk::Dense { .. } => "dense".into(),
        PeriodicJk::RsGdf { auxbasis, .. } => format!("rsgdf (aux: {auxbasis})"),
    }
}

fn budget_bytes(plan: &PeriodicPlan) -> Option<usize> {
    match &plan.jk {
        PeriodicJk::Dense { .. } => None,
        PeriodicJk::RsGdf { budget_bytes, .. } => *budget_bytes,
    }
}

fn max_eri_bytes(plan: &PeriodicPlan) -> usize {
    match &plan.jk {
        PeriodicJk::Dense { max_eri_bytes } => *max_eri_bytes,
        PeriodicJk::RsGdf { .. } => ferric_pbc::dense_aft::DEFAULT_DENSE_AFT_MAX_BYTES,
    }
}

/// The header every periodic printout starts with: method, cell, mesh.
fn print_header(cfg: &Config, plan: &PeriodicPlan, s: &Setup) {
    let method = match &plan.functional {
        Some(f) => format!("{}[{f}]", plan.route.label()),
        None => plan.route.label().to_string(),
    };
    let point = if plan.kmesh.is_some() {
        "k-point"
    } else {
        "Gamma point"
    };
    println!(
        "Periodic {method}/{} on {} ({point})",
        s.bs.name, cfg.molecule.xyz
    );
    println!(
        "  lattice    (Bohr, rows{})",
        if plan.unit_bohr {
            ""
        } else {
            "; converted from Angstrom"
        }
    );
    for r in &plan.lattice_bohr {
        println!("               [{:14.8} {:14.8} {:14.8}]", r[0], r[1], r[2]);
    }
    println!("  volume     = {:.6} Bohr^3", s.cell.volume());
    match plan.kmesh {
        Some((n, c)) => println!(
            "  kmesh      = {}x{}x{} ({}, nk = {})",
            n[0],
            n[1],
            n[2],
            match c {
                MeshCentring::Gamma => "Gamma-centred",
                MeshCentring::MonkhorstPack => "Monkhorst-Pack",
            },
            n[0] * n[1] * n[2]
        ),
        None => println!("  kmesh      = Gamma point only"),
    }
    println!("  exxdiv     = {}", exx_name(plan.exxdiv));
    println!("  jk         = {}", jk_name(plan));
    println!("  nbasis     = {}", s.prep.nbasis());
}

fn print_madelung(plan: &PeriodicPlan, v_m: f64) {
    println!(
        "  madelung   = {:.10} ({})",
        v_m,
        match plan.exxdiv {
            ExxDiv::Ewald => "applied",
            ExxDiv::None => "not applied, exxdiv = none",
        }
    );
}

fn warn_unconverged(what: &str, iterations: usize) {
    eprintln!(
        "warning: the periodic {what} SCF did not converge in {iterations} iterations — the \
         energy above must not be quoted"
    );
}

// ─────────────────────────────────────────────────────── Gamma integrals ──

fn gamma_scf_config(plan: &PeriodicPlan) -> RhfConfig {
    RhfConfig {
        use_sad_guess: false,
        density_conv: plan.density_conv,
        max_iter: plan.max_iter,
        ..Default::default()
    }
}

/// The Gamma J/K (and ov-integral) source, built with the run's `exxdiv`.
enum GammaInts {
    Dense(Box<DenseAftEri>),
    RsGdf(Box<RsGdf>),
}

impl GammaInts {
    fn scf(&self) -> GammaUhfIntegrals<'_> {
        match self {
            GammaInts::Dense(e) => GammaUhfIntegrals::DenseAft(e),
            GammaInts::RsGdf(g) => GammaUhfIntegrals::RsGdf(g),
        }
    }
    fn mp2(&self) -> GammaMp2Integrals<'_> {
        match self {
            GammaInts::Dense(e) => GammaMp2Integrals::DenseAft(e),
            GammaInts::RsGdf(g) => GammaMp2Integrals::RsGdf(g),
        }
    }
    fn drpa(&self) -> GammaDrpaIntegrals<'_> {
        match self {
            GammaInts::Dense(e) => GammaDrpaIntegrals::DenseAft(e),
            GammaInts::RsGdf(g) => GammaDrpaIntegrals::RsGdf(g),
        }
    }
}

struct GammaSystem {
    hc: PeriodicHcore,
    ints: GammaInts,
}

fn gamma_system(plan: &PeriodicPlan, s: &Setup) -> Result<GammaSystem, FerricError> {
    let hc = periodic_hcore(
        &s.cell,
        &s.prep,
        &PeriodicHcoreConfig::with_omega(s.omega_bohr),
    )?;
    let ints = match &s.aux {
        None => GammaInts::Dense(Box::new(DenseAftEri::build(
            &s.cell,
            &s.prep,
            &hc.s,
            plan.exxdiv,
            DEFAULT_DENSE_AFT_PRECISION,
            max_eri_bytes(plan),
        )?)),
        Some(aux) => {
            let cfg = RsGdfConfig {
                exxdiv: plan.exxdiv,
                budget_bytes: budget_bytes(plan),
                ..Default::default()
            };
            GammaInts::RsGdf(Box::new(RsGdf::build(&s.cell, &s.prep, aux, &hc.s, &cfg)?))
        }
    };
    Ok(GammaSystem { hc, ints })
}

/// Closed-shell Gamma RHF on injected J/K (the `run_rhf_gamma` assembly).
fn inject_rhf<'a>(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    cfg: &RhfConfig,
    j: Box<dyn ferric_scf::fock::JBuilder + 'a>,
    k: Box<dyn ferric_scf::fock::KBuilder + 'a>,
) -> Result<ScfResult, FerricError> {
    let inj = ferric_scf::rhf::PeriodicInjection {
        s: hc.s.clone(),
        h: hc.h.clone(),
        vnn: hc.enn,
        j,
        k,
        xc: None,
    };
    let op = Operator::coulomb();
    // Never read on the injected path; required by the signature.
    let bounds = SchwarzBounds::compute(op, prep)?;
    let ctx = ParallelContext::default();
    ferric_scf::rhf::solve_rhf_injected(&ctx, cell.mol(), prep, op, &bounds, cfg, inj)
}

/// Gamma RHF on `sys` (not required to converge; the caller decides).
fn gamma_rhf_scf(s: &Setup, sys: &GammaSystem, cfg: &RhfConfig) -> Result<ScfResult, FerricError> {
    match &sys.ints {
        GammaInts::Dense(e) => inject_rhf(
            &s.cell,
            &s.prep,
            &sys.hc,
            cfg,
            Box::new(e.j_builder()),
            Box::new(e.k_builder()),
        ),
        GammaInts::RsGdf(g) => inject_rhf(
            &s.cell,
            &s.prep,
            &sys.hc,
            cfg,
            Box::new(g.j_builder()),
            Box::new(g.k_builder()),
        ),
    }
}

// ───────────────────────────────────────────────────────── Gamma SCF ──

/// Gamma RHF: `(scf, E_nn, v_M)`.
fn gamma_rhf_driver(plan: &PeriodicPlan, s: &Setup) -> Result<(ScfResult, f64, f64), FerricError> {
    let sys = gamma_system(plan, s)?;
    let r = gamma_rhf_scf(s, &sys, &gamma_scf_config(plan))?;
    // Reported for both exxdiv settings (the builders only store it for
    // Ewald), as `run_rhf_gamma` does.
    let v_m = ferric_pbc::ewald::madelung_constant(&s.cell)?;
    Ok((r, sys.hc.enn, v_m))
}

fn run_gamma_rhf(cfg: &Config, plan: &PeriodicPlan, s: &Setup) -> Outcome {
    let (r, enn, v_m) = gamma_rhf_driver(plan, s).unwrap_or_else(|e| die(e));
    print_header(cfg, plan, s);
    println!("  iterations = {}", r.iterations);
    println!("  converged  = {}", r.converged);
    println!("  e_nuc      = {:.10} Hartree/cell (Ewald)", enn);
    print_madelung(plan, v_m);
    println!("  energy     = {:.10} Hartree/cell", r.energy);
    if !r.converged {
        warn_unconverged("RHF", r.iterations);
    }
    Outcome {
        energy: r.energy,
        converged: r.converged,
        energy_is: "total",
    }
}

fn periodic_grid(plan: &PeriodicPlan) -> PeriodicGridConfig {
    PeriodicGridConfig {
        neighbour_cutoff: plan.neighbour_cutoff_bohr,
        ..PeriodicGridConfig::with_size(plan.n_radial, plan.n_angular)
    }
}

/// Gamma RKS: `(result, E_nn)`.
fn gamma_rks_driver(
    plan: &PeriodicPlan,
    s: &Setup,
) -> Result<(ferric_pbc::GammaRksResult, f64), FerricError> {
    let functional = plan.functional.as_deref().unwrap_or("LDA");
    let sys = gamma_system(plan, s)?;
    let mut c = GammaRksConfig::new(functional);
    (c.grid, c.exxdiv, c.scf) = (periodic_grid(plan), plan.exxdiv, gamma_scf_config(plan));
    let r = gamma_rks(&s.cell, &s.prep, &sys.hc, sys.ints.scf(), &c)?;
    Ok((r, sys.hc.enn))
}

fn run_gamma_rks(cfg: &Config, plan: &PeriodicPlan, s: &Setup) -> Outcome {
    let (r, enn) = gamma_rks_driver(plan, s).unwrap_or_else(|e| die(e));
    print_header(cfg, plan, s);
    println!(
        "  grid       = {} points ({}x{}, neighbour cutoff {:.4} Bohr), {:.8} electrons",
        r.n_grid_points, plan.n_radial, plan.n_angular, r.neighbour_cutoff, r.electrons_on_grid
    );
    println!("  iterations = {}", r.scf.iterations);
    println!("  converged  = {}", r.scf.converged);
    println!("  e_nuc      = {:.10} Hartree/cell (Ewald)", enn);
    print_madelung(plan, r.madelung);
    println!(
        "  E_xc       = {:.10} Hartree/cell (exact exchange a_x = {})",
        r.e_xc, r.exact_exchange_fraction
    );
    println!("  energy     = {:.10} Hartree/cell", r.scf.energy);
    if !r.scf.converged {
        warn_unconverged("RKS", r.scf.iterations);
    }
    Outcome {
        energy: r.scf.energy,
        converged: r.scf.converged,
        energy_is: "total",
    }
}

/// The common part of every Gamma open-shell driver's result.
struct OpenParts {
    scf: ScfResult,
    none_stage: Option<ScfResult>,
    madelung: f64,
    nocc: (usize, usize),
    s2: f64,
    gap_alpha: Option<f64>,
    gap_beta: Option<f64>,
    gaps_satisfied: bool,
    /// KS only: (E_xc, a_x, grid points, electrons on grid).
    ks: Option<(f64, f64, Option<(usize, f64)>)>,
}

fn gamma_open_driver(plan: &PeriodicPlan, s: &Setup) -> Result<(OpenParts, f64), FerricError> {
    let sys = gamma_system(plan, s)?;
    let ints = sys.ints.scf();
    let scf = gamma_scf_config(plan);
    let f = plan.functional.clone().unwrap_or_default();
    let parts = match plan.route {
        PeriodicRoute::Uhf => {
            let c = GammaUhfConfig {
                exxdiv: plan.exxdiv,
                ewald_start: plan.ewald_start,
                scf,
                initial_mos: None,
            };
            let r = gamma_uhf(&s.cell, &s.prep, &sys.hc, ints, &c)?;
            OpenParts {
                gap_alpha: r.gaps.gap_alpha,
                gap_beta: r.gaps.gap_beta,
                gaps_satisfied: r.gaps.satisfied(),
                scf: r.scf,
                none_stage: r.none_stage,
                madelung: r.madelung,
                nocc: r.nocc,
                s2: r.s2,
                ks: None,
            }
        }
        PeriodicRoute::Rohf => {
            let c = GammaRohfConfig {
                exxdiv: plan.exxdiv,
                ewald_start: plan.ewald_start,
                scf,
                initial_mos: None,
            };
            let r = gamma_rohf(&s.cell, &s.prep, &sys.hc, ints, &c)?;
            OpenParts {
                gap_alpha: r.gaps.gap_alpha,
                gap_beta: r.gaps.gap_beta,
                gaps_satisfied: r.gaps.satisfied(),
                scf: r.scf,
                none_stage: r.none_stage,
                madelung: r.madelung,
                nocc: r.nocc,
                s2: r.s2,
                ks: None,
            }
        }
        PeriodicRoute::Uks => {
            let mut c = GammaUksConfig::new(&f);
            (c.grid, c.exxdiv, c.ewald_start) =
                (periodic_grid(plan), plan.exxdiv, plan.ewald_start);
            c.scf = scf;
            let r = gamma_uks(&s.cell, &s.prep, &sys.hc, ints, &c)?;
            OpenParts {
                gap_alpha: r.gaps.gap_alpha,
                gap_beta: r.gaps.gap_beta,
                gaps_satisfied: r.gaps.satisfied(),
                ks: Some((
                    r.e_xc,
                    r.exact_exchange_fraction,
                    r.grid
                        .as_ref()
                        .map(|g| (g.n_grid_points, g.electrons_on_grid)),
                )),
                scf: r.scf,
                none_stage: r.none_stage,
                madelung: r.madelung,
                nocc: r.nocc,
                s2: r.s2,
            }
        }
        PeriodicRoute::Roks => {
            let mut c = GammaRoksConfig::new(&f);
            (c.grid, c.exxdiv, c.ewald_start) =
                (periodic_grid(plan), plan.exxdiv, plan.ewald_start);
            // Keep GammaRoksConfig::new's hybrid level shift (and its 600
            // cap unless [scf] max_iter was written): a DIIS robustness
            // default that vanishes at convergence (see its doc).
            c.scf = RhfConfig {
                level_shift: c.scf.level_shift,
                max_iter: effective_max_iter(plan),
                ..scf
            };
            let r = gamma_roks(&s.cell, &s.prep, &sys.hc, ints, &c)?;
            OpenParts {
                gap_alpha: r.gaps.gap_alpha,
                gap_beta: r.gaps.gap_beta,
                gaps_satisfied: r.gaps.satisfied(),
                ks: Some((
                    r.e_xc,
                    r.exact_exchange_fraction,
                    r.grid
                        .as_ref()
                        .map(|g| (g.n_grid_points, g.electrons_on_grid)),
                )),
                scf: r.scf,
                none_stage: r.none_stage,
                madelung: r.madelung,
                nocc: r.nocc,
                s2: r.s2,
            }
        }
        other => {
            return Err(FerricError::General(format!(
                "internal: {} is not an open-shell periodic route",
                other.label()
            )))
        }
    };
    Ok((parts, sys.hc.enn))
}

fn fmt_gap(g: Option<f64>) -> String {
    g.map_or_else(|| "n/a".into(), |v| format!("{v:.6}"))
}

fn ewald_start_name(plan: &PeriodicPlan) -> &'static str {
    match (plan.exxdiv, plan.ewald_start) {
        (ExxDiv::None, _) => "n/a (exxdiv = none)",
        (ExxDiv::Ewald, ferric_pbc::EwaldStart::Staged) => "staged",
        (ExxDiv::Ewald, ferric_pbc::EwaldStart::Direct) => "direct",
    }
}

fn run_gamma_open(cfg: &Config, plan: &PeriodicPlan, s: &Setup) -> Outcome {
    let (p, enn) = gamma_open_driver(plan, s).unwrap_or_else(|e| die(e));
    print_header(cfg, plan, s);
    println!(
        "  mult       = {} (nalpha={}, nbeta={} per cell)",
        cfg.molecule.multiplicity, p.nocc.0, p.nocc.1
    );
    println!("  ewald_start= {}", ewald_start_name(plan));
    if let Some((_, _, Some((npts, nel)))) = p.ks {
        println!(
            "  grid       = {npts} points ({}x{}), {nel:.8} electrons",
            plan.n_radial, plan.n_angular
        );
    }
    if let Some(ns) = &p.none_stage {
        println!(
            "  none stage = {:.10} Hartree/cell ({} iterations, exxdiv = none)",
            ns.energy, ns.iterations
        );
    }
    println!("  iterations = {}", p.scf.iterations);
    println!("  converged  = {}", p.scf.converged);
    println!("  e_nuc      = {:.10} Hartree/cell (Ewald)", enn);
    print_madelung(plan, p.madelung);
    if let Some((e_xc, a_x, _)) = p.ks {
        println!("  E_xc       = {e_xc:.10} Hartree/cell (exact exchange a_x = {a_x})");
    }
    println!("  energy     = {:.10} Hartree/cell", p.scf.energy);
    println!("  <S^2>      = {:.6}", p.s2);
    println!(
        "  gaps       = alpha {} / beta {} Hartree ({})",
        fmt_gap(p.gap_alpha),
        fmt_gap(p.gap_beta),
        if p.gaps_satisfied {
            "every gap >= the applied Madelung shift"
        } else {
            "a gap is BELOW the applied Madelung shift: possible Ewald-trap state"
        }
    );
    if !p.gaps_satisfied {
        eprintln!(
            "warning: a per-spin HOMO-LUMO gap is below the applied Madelung shift (the Gamma \
             Ewald-trap diagnostic); inspect the state before quoting this energy"
        );
    }
    if !p.scf.converged {
        warn_unconverged(plan.route.label(), p.scf.iterations);
    }
    Outcome {
        energy: p.scf.energy,
        converged: p.scf.converged,
        energy_is: "total",
    }
}

// ─────────────────────────────────────────────── Gamma MP2 / dRPA ──

fn den_name(d: Mp2Denominators) -> &'static str {
    match d {
        Mp2Denominators::MadelungShifted => "shifted",
        Mp2Denominators::Unshifted => "unshifted",
    }
}

fn denominators(plan: &PeriodicPlan) -> Mp2Denominators {
    plan.denominators
        .unwrap_or_else(|| die("internal: periodic correlation without [cell] denominators"))
}

/// The numbers both Gamma correlation drivers return.
struct CorrParts {
    corr: f64,
    total: f64,
    os_ss: Option<(f64, f64)>,
    madelung: f64,
    occ_shift: f64,
    nocc_active: usize,
    nvir: usize,
    naux: Option<usize>,
    quad_points: Option<usize>,
}

fn gamma_corr_driver(
    plan: &PeriodicPlan,
    s: &Setup,
) -> Result<(ScfResult, CorrParts), FerricError> {
    let sys = gamma_system(plan, s)?;
    let rhf = gamma_rhf_scf(s, &sys, &gamma_scf_config(plan))?;
    if !rhf.converged {
        return Err(FerricError::Convergence(format!(
            "the Gamma RHF reference did not converge in {} iterations (last E = {})",
            rhf.iterations, rhf.energy
        )));
    }
    let den = denominators(plan);
    let c = if plan.route == PeriodicRoute::Mp2 {
        let c = GammaMp2Config {
            frozen_core: plan.frozen_core,
            reference_exxdiv: plan.exxdiv,
            denominators: den,
            budget_bytes: budget_bytes(plan),
        };
        let r = gamma_mp2(&s.cell, &rhf, sys.ints.mp2(), &c)?;
        CorrParts {
            corr: r.mp2_corr,
            total: r.total_energy,
            os_ss: Some((r.components.e_os, r.components.e_ss)),
            madelung: r.madelung,
            occ_shift: r.occ_shift,
            nocc_active: r.nocc_active,
            nvir: r.nvir,
            naux: r.naux,
            quad_points: None,
        }
    } else {
        let c = GammaDrpaConfig {
            frozen_core: plan.frozen_core,
            reference_exxdiv: plan.exxdiv,
            denominators: den,
            quad_points: plan.quad_points.unwrap_or(DEFAULT_GAMMA_DRPA_QUAD_POINTS),
            budget_bytes: budget_bytes(plan),
        };
        let r = gamma_drpa(&s.cell, &rhf, sys.ints.drpa(), &c)?;
        CorrParts {
            corr: r.drpa_corr,
            total: r.total_energy,
            os_ss: None,
            madelung: r.madelung,
            occ_shift: r.occ_shift,
            nocc_active: r.nocc_active,
            nvir: r.nvir,
            naux: r.naux,
            quad_points: r.quad_points,
        }
    };
    Ok((rhf, c))
}

fn run_gamma_corr(cfg: &Config, plan: &PeriodicPlan, s: &Setup) -> Outcome {
    let (rhf, c) = gamma_corr_driver(plan, s).unwrap_or_else(|e| die(e));
    let label = plan.route.label();
    print_header(cfg, plan, s);
    println!(
        "  denominators = {} (occupied shift {:.10})",
        den_name(denominators(plan)),
        c.occ_shift
    );
    println!(
        "  nocc_active = {}, nvir = {}{}{}",
        c.nocc_active,
        c.nvir,
        c.naux.map_or(String::new(), |n| format!(", naux = {n}")),
        c.quad_points
            .map_or(String::new(), |q| format!(", quad_points = {q}"))
    );
    print_madelung(plan, c.madelung);
    println!("  RHF energy = {:.10} Hartree/cell", rhf.energy);
    println!("  SCF iters  = {}", rhf.iterations);
    if let Some((os, ss)) = c.os_ss {
        println!("  E_os       = {os:.10} Hartree/cell");
        println!("  E_ss       = {ss:.10} Hartree/cell");
    }
    println!("  {label} corr   = {:.10} Hartree/cell", c.corr);
    println!("  Total      = {:.10} Hartree/cell", c.total);
    Outcome {
        energy: c.total,
        converged: rhf.converged,
        energy_is: "correlated_total",
    }
}

// ─────────────────────────────────────────────────────── k-point SCF ──

fn k_mesh(plan: &PeriodicPlan, s: &Setup) -> KPointMesh {
    let Some((n, c)) = plan.kmesh else {
        die("internal: k-point route without kmesh");
    };
    KPointMesh::new(&s.cell, n, c).unwrap_or_else(|e| die(format!("[cell] kmesh: {e}")))
}

fn kscf_config(plan: &PeriodicPlan) -> KScfConfig {
    KScfConfig {
        max_iter: plan.max_iter,
        energy_conv: plan.energy_conv,
        grad_conv: plan.grad_conv,
        ..Default::default()
    }
}

fn kdense_config(plan: &PeriodicPlan) -> KDenseAftConfig {
    KDenseAftConfig {
        max_bytes: max_eri_bytes(plan),
        ..Default::default()
    }
}

fn krsgdf_config(plan: &PeriodicPlan) -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: RsGdfConfig {
            budget_bytes: budget_bytes(plan),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn kjk_kind(s: &Setup) -> KJkKind {
    if s.aux.is_some() {
        KJkKind::RsGdf
    } else {
        KJkKind::Dense
    }
}

fn print_lindep(l: &ferric_pbc::LindepReport) {
    println!(
        "  lindep     = threshold {:.1e}, kept {}..{} per k ({} total){}",
        l.threshold,
        l.min_kept,
        l.max_kept,
        l.total_kept,
        if l.near_noise_floor {
            ", NEAR the overlap noise floor"
        } else {
            ""
        }
    );
}

/// k-point RHF: `(result, mesh Madelung constant)`.
fn krhf_driver(
    plan: &PeriodicPlan,
    s: &Setup,
    mesh: &KPointMesh,
) -> Result<(KScfResult, f64), FerricError> {
    let mut c = KRhfConfig::for_cell(&s.cell, plan.exxdiv);
    c.scf = kscf_config(plan);
    c.hcore = PeriodicHcoreConfig::with_omega(s.omega_bohr);
    (c.jk, c.dense, c.rsgdf) = (kjk_kind(s), kdense_config(plan), krsgdf_config(plan));
    let r = solve_krhf(&s.cell, &s.prep, s.aux.as_ref(), mesh, &c)?;
    let v_m = mesh.madelung(&s.cell)?;
    Ok((r, v_m))
}

fn run_krhf(cfg: &Config, plan: &PeriodicPlan, s: &Setup) -> Outcome {
    let mesh = k_mesh(plan, s);
    let (r, v_m) = krhf_driver(plan, s, &mesh).unwrap_or_else(|e| die(e));
    print_header(cfg, plan, s);
    print_lindep(&r.lindep);
    println!("  iterations = {}", r.iterations);
    println!("  converged  = {}", r.converged);
    println!("  e_nuc      = {:.10} Hartree/cell (Ewald)", r.e_nuc);
    print_madelung(plan, v_m);
    println!(
        "  HOMO/LUMO  = {:.6} / {:.6} Hartree (over the mesh)",
        r.homo, r.lumo
    );
    println!("  energy     = {:.10} Hartree/cell", r.energy);
    if !r.converged {
        warn_unconverged("k-point RHF", r.iterations);
    }
    Outcome {
        energy: r.energy,
        converged: r.converged,
        energy_is: "total",
    }
}

fn kuhf_driver(
    plan: &PeriodicPlan,
    s: &Setup,
    mesh: &KPointMesh,
) -> Result<ferric_pbc::KUhfResult, FerricError> {
    let mut c = KUhfConfig::for_cell(&s.cell, plan.exxdiv);
    (c.scf, c.ewald_start) = (kscf_config(plan), plan.ewald_start);
    c.hcore = PeriodicHcoreConfig::with_omega(s.omega_bohr);
    (c.jk, c.dense, c.rsgdf) = (kjk_kind(s), kdense_config(plan), krsgdf_config(plan));
    c.budget_bytes = budget_bytes(plan);
    solve_kuhf(&s.cell, &s.prep, s.aux.as_ref(), mesh, &c)
}

fn run_kuhf(cfg: &Config, plan: &PeriodicPlan, s: &Setup) -> Outcome {
    let mesh = k_mesh(plan, s);
    let r = kuhf_driver(plan, s, &mesh).unwrap_or_else(|e| die(e));
    let u = &r.scf;
    print_header(cfg, plan, s);
    println!(
        "  mult       = {} (nalpha={}, nbeta={} per cell)",
        cfg.molecule.multiplicity, r.nocc.0, r.nocc.1
    );
    println!("  ewald_start= {}", ewald_start_name(plan));
    print_lindep(&u.lindep);
    if let Some(ns) = &r.none_stage {
        println!(
            "  none stage = {:.10} Hartree/cell ({} iterations, exxdiv = none)",
            ns.energy, ns.iterations
        );
    }
    println!("  iterations = {}", u.iterations);
    println!("  converged  = {}", u.converged);
    println!("  e_nuc      = {:.10} Hartree/cell (Ewald)", u.e_nuc);
    print_madelung(plan, r.madelung);
    println!("  energy     = {:.10} Hartree/cell", u.energy);
    println!(
        "  <S^2>      = {:.6} (of the giant supercell determinant, not per cell)",
        u.s2
    );
    println!(
        "  gaps       = alpha {} / beta {} Hartree",
        fmt_gap(r.gaps.gap_alpha),
        fmt_gap(r.gaps.gap_beta)
    );
    if !u.converged {
        warn_unconverged("k-point UHF", u.iterations);
    }
    Outcome {
        energy: u.energy,
        converged: u.converged,
        energy_is: "total",
    }
}

// ─────────────────────────────────────────────── k-point MP2 / dRPA ──

/// k-point integrals kept alive past the SCF for the correlation step.
enum KInts {
    Dense(Box<KDenseAftPairs>),
    RsGdf(Box<KRsGdf>),
}

impl KInts {
    fn corr(&self) -> KCorrIntegrals<'_> {
        match self {
            KInts::Dense(p) => KCorrIntegrals::DenseAft(p),
            KInts::RsGdf(g) => KCorrIntegrals::RsGdf(g),
        }
    }
}

fn require_kscf_converged(r: &KScfResult) -> Result<(), FerricError> {
    if r.converged {
        return Ok(());
    }
    Err(FerricError::Convergence(format!(
        "the k-point RHF reference did not converge in {} iterations (last E = {})",
        r.iterations, r.energy
    )))
}

/// k-point RHF whose integrals outlive the SCF (dense: the SCF kernels are
/// dropped and the correlation pair tensors built; rsgdf: one object).
fn krhf_with_ints(
    plan: &PeriodicPlan,
    s: &Setup,
    mesh: &KPointMesh,
) -> Result<(KScfResult, KInts), FerricError> {
    let scf = kscf_config(plan);
    let hk = periodic_hcore_kpts(
        &s.cell,
        &s.prep,
        mesh,
        &PeriodicHcoreConfig::with_omega(s.omega_bohr),
    )?;
    match &s.aux {
        None => {
            let kc = kdense_config(plan);
            let eri = KDenseAftEri::build(&s.cell, &s.prep, mesh, &hk.s, plan.exxdiv, &kc)?;
            let inj = KPointInjection {
                s: hk.s,
                h: hk.h,
                vnn: hk.enn,
                jk: Box::new(eri.jk_builder()),
            };
            let r = solve_krhf_injected(&s.cell, mesh, &scf, inj)?;
            drop(eri);
            require_kscf_converged(&r)?;
            let pairs = KDenseAftPairs::build(&s.cell, &s.prep, mesh, &kc)?;
            Ok((r, KInts::Dense(Box::new(pairs))))
        }
        Some(aux) => {
            let gdf = KRsGdf::build(&s.cell, &s.prep, aux, mesh, &hk.s, &krsgdf_config(plan))?
                .with_exxdiv(plan.exxdiv);
            let inj = KPointInjection {
                s: hk.s,
                h: hk.h,
                vnn: hk.enn,
                jk: Box::new(gdf.jk_builder()),
            };
            let r = solve_krhf_injected(&s.cell, mesh, &scf, inj)?;
            require_kscf_converged(&r)?;
            Ok((r, KInts::RsGdf(Box::new(gdf))))
        }
    }
}

/// What the k-point MP2 / dRPA drivers report.
struct KCorrOut {
    corr: f64,
    total: f64,
    madelung: f64,
    occ_shift: f64,
    nocc_active: usize,
    /// Virtuals per k.
    nvir: Vec<usize>,
    /// Method-specific printout lines.
    extra: Vec<String>,
}

fn kcorr_driver(
    plan: &PeriodicPlan,
    s: &Setup,
    mesh: &KPointMesh,
    den: Mp2Denominators,
) -> Result<(KScfResult, KCorrOut), FerricError> {
    let (r, ints) = krhf_with_ints(plan, s, mesh)?;
    let out = if plan.route == PeriodicRoute::Mp2 {
        let c = KMp2Config {
            frozen_core: plan.frozen_core,
            reference_exxdiv: plan.exxdiv,
            denominators: den,
            budget_bytes: budget_bytes(plan),
            mutation: None,
        };
        let m = kpoint_mp2(&s.cell, mesh, &r, ints.corr(), &c)?;
        KCorrOut {
            extra: vec![
                format!("  E_os       = {:.10} Hartree/cell", m.e_os),
                format!("  E_ss       = {:.10} Hartree/cell", m.e_ss),
                format!("  E_direct   = {:.10} Hartree/cell", m.e_direct),
            ],
            corr: m.mp2_corr,
            total: m.total_energy,
            madelung: m.madelung,
            occ_shift: m.occ_shift,
            nocc_active: m.nocc_active,
            nvir: m.nvir,
        }
    } else {
        let q = plan.quad_points.unwrap_or(DEFAULT_KDRPA_QUAD_POINTS);
        let c = KDrpaConfig {
            frozen_core: plan.frozen_core,
            reference_exxdiv: plan.exxdiv,
            denominators: den,
            quad_points: q,
            energy: plan.drpa_energy,
            budget_bytes: budget_bytes(plan),
            mutation: None,
        };
        let d = kpoint_drpa(&s.cell, mesh, &r, ints.corr(), &c)?;
        let energy = match plan.drpa_energy {
            KDrpaEnergy::Quadrature => format!("quadrature, {q} points"),
            KDrpaEnergy::Plasmon => "plasmon".into(),
            KDrpaEnergy::SecondOrder => "second-order".into(),
        };
        KCorrOut {
            extra: vec![
                format!("  dRPA energy = {energy}"),
                format!("  per-q      = {:?}", d.per_q),
            ],
            corr: d.drpa_corr,
            total: d.total_energy,
            madelung: d.madelung,
            occ_shift: d.occ_shift,
            nocc_active: d.nocc_active,
            nvir: d.nvir,
        }
    };
    Ok((r, out))
}

fn run_kcorr(cfg: &Config, plan: &PeriodicPlan, s: &Setup) -> Outcome {
    let mesh = k_mesh(plan, s);
    let den = denominators(plan);
    let label = plan.route.label();
    let (r, k) = kcorr_driver(plan, s, &mesh, den).unwrap_or_else(|e| die(e));
    print_header(cfg, plan, s);
    print_lindep(&r.lindep);
    println!(
        "  denominators = {} (occupied shift {:.10})",
        den_name(den),
        k.occ_shift
    );
    println!(
        "  nocc_active = {}, nvir per k = {:?}",
        k.nocc_active, k.nvir
    );
    print_madelung(plan, k.madelung);
    println!("  RHF energy = {:.10} Hartree/cell", r.energy);
    println!("  SCF iters  = {}", r.iterations);
    for line in &k.extra {
        println!("{line}");
    }
    println!("  {label} corr   = {:.10} Hartree/cell", k.corr);
    println!("  Total      = {:.10} Hartree/cell", k.total);
    Outcome {
        energy: k.total,
        converged: r.converged,
        energy_is: "correlated_total",
    }
}

// ───────────────────────────────────────── Gamma RHF optimization ──

/// `task = "optimize"` for Gamma RHF on the dense-AFT J/K: the analytic
/// periodic force (`ferric_pbc::gamma_rhf_gradient`) fed to ferric's
/// Cartesian BFGS (`optimize_coordinates`) at a FIXED lattice. Each step
/// rebuilds the cell, basis, hcore and dense tensor at the new positions and
/// runs a fresh SCF, which must converge (a gradient is only meaningful at a
/// stationary density). No stress / lattice relaxation.
fn run_gamma_rhf_optimize(cfg: &Config, plan: &PeriodicPlan, s: &Setup) -> Outcome {
    use ferric_scf::optimize::{optimize_coordinates, CoordSystem, OptimizeConfig};
    let opt_config = OptimizeConfig {
        max_steps: cfg.optimize.max_steps.unwrap_or(100),
        g_max_thresh: cfg.optimize.g_max_thresh.unwrap_or(4.5e-4),
        g_rms_thresh: cfg.optimize.g_rms_thresh.unwrap_or(3.0e-4),
        e_conv: cfg.optimize.e_conv.unwrap_or(1e-6),
        trust_radius: cfg.optimize.trust_radius.unwrap_or(0.1),
        // Cartesian only; `periodic_plan` refused anything else.
        coord_system: CoordSystem::Cartesian,
    };
    let base = s.cell.mol().clone();
    let x0: Vec<f64> = base.atoms.iter().flat_map(|a| [a.x, a.y, a.zpos]).collect();
    // The Ewald split is a numerical knob of the fixed lattice; hold it fixed
    // so every step's energy and gradient share one hcore convention.
    let omega = s.omega_bohr;
    let scf_cfg = gamma_scf_config(plan);
    let max_bytes = max_eri_bytes(plan);
    let at = |x: &[f64]| -> Molecule {
        let mut m = base.clone();
        for (i, a) in m.atoms.iter_mut().enumerate() {
            (a.x, a.y, a.zpos) = (x[3 * i], x[3 * i + 1], x[3 * i + 2]);
        }
        m
    };
    let energy_and_gradient = |x: &[f64]| -> Result<(f64, Vec<f64>), FerricError> {
        let cell = Cell::new(at(x), plan.lattice_bohr)?;
        let prep = PreparedBasis::new(cell.mol(), &s.bs)?;
        let hcfg = PeriodicHcoreConfig::with_omega(omega);
        let hc = periodic_hcore(&cell, &prep, &hcfg)?;
        let eri = DenseAftEri::build(
            &cell,
            &prep,
            &hc.s,
            plan.exxdiv,
            DEFAULT_DENSE_AFT_PRECISION,
            max_bytes,
        )?;
        let scf = inject_rhf(
            &cell,
            &prep,
            &hc,
            &scf_cfg,
            Box::new(eri.j_builder()),
            Box::new(eri.k_builder()),
        )?;
        if !scf.converged {
            return Err(FerricError::Convergence(format!(
                "the Gamma RHF SCF did not converge in {} iterations during the optimization \
                 (last E = {}); a gradient at a non-stationary density is meaningless",
                scf.iterations, scf.energy
            )));
        }
        let g = gamma_rhf_gradient(&cell, &prep, &hcfg, &hc, &eri, &scf, plan.exxdiv)?;
        Ok((scf.energy, g.iter().copied().collect()))
    };
    print_header(cfg, plan, s);
    println!("  task       = optimize (Cartesian BFGS, fixed lattice)");
    let (x, energy, steps, converged) =
        optimize_coordinates(&x0, &opt_config, energy_and_gradient).unwrap_or_else(|e| die(e));
    let fin = at(&x);
    println!("  steps      = {steps}");
    println!("  converged  = {converged}");
    println!("  energy     = {energy:.10} Hartree/cell");
    println!("  final geometry (Angstrom, Cartesian; atoms may sit outside the reference cell):");
    for a in &fin.atoms {
        println!(
            "    {:<3} {:14.8} {:14.8} {:14.8}",
            a.symbol,
            a.x / ANGSTROM_TO_BOHR,
            a.y / ANGSTROM_TO_BOHR,
            a.zpos / ANGSTROM_TO_BOHR
        );
    }
    if !converged {
        eprintln!("warning: the periodic geometry optimization did not converge in {steps} steps");
    }
    Outcome {
        energy,
        converged,
        energy_is: "total",
    }
}
