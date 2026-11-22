//! `method.kind = "casscf"` and `"nevpt2"`: two-step CASSCF and SC-NEVPT2
//! through `ferric_ci::driver`, the code the Python `run_casscf` and
//! `run_nevpt2` run.
//!
//! The run is the exact-J/K Coulomb RHF start with the library defaults, then
//! the CAS step, so the CLI reads only `[cas]` (active space and orbital
//! optimization), `[cas.ci]` (the CI eigensolver), `[scf]` `max_iter` /
//! `energy_conv` / `density_conv`, `[memory]`, `[parallel]` and `[output]`.
//! [`resolve_plan`] refuses every other section and key at load time, so no
//! key is read by one surface and ignored by the other.

use crate::config::Config;
use crate::error::die;
use ferric_ci::casscf::{CasScfIntegrals, CasScfOptimizer, CasScfStabilityMode};
use ferric_ci::driver::{casscf_on_rhf, nevpt2_on_rhf, rhf_start, Nevpt2On, ScfStart};
use ferric_ci::{CasScfConfig, Nevpt2Classes};
use ferric_core::basis;
use ferric_core::certificate::BuildStamp;
use ferric_core::mol::Molecule;
use ferric_scf::{EcpMolecule, PrepareOptions};
use serde::Deserialize;

/// `[cas]`: the active space of `method.kind = "casscf"` / `"nevpt2"` and the
/// CASSCF orbital optimization. Python `run_casscf` / `run_nevpt2` take the
/// same settings (`n_active`, `n_elec_active`, `n_core`, `reference`,
/// `canonicalize`, `casscf_grad_tol`, `casscf_energy_tol`, `casscf_max_iter`,
/// `casscf_max_step`, `casscf_integrals`, `casscf_integral_thresh`,
/// `casscf_auxbasis`, `casscf_optimizer`, `casscf_stability`).
#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct CasCfg {
    /// Active spatial orbitals.
    pub n_active: usize,
    /// Active electrons `[n_alpha, n_beta]`; their difference sets the spin
    /// of the CAS state.
    pub n_elec_active: [usize; 2],
    /// Doubly occupied core orbitals below the active ones. Default: the
    /// electrons the active space leaves, halved.
    pub n_core: Option<usize>,
    /// `nevpt2` only: `"casscf"` (default) or `"casci"` (CAS-CI on the RHF
    /// orbitals).
    pub reference: Option<String>,
    /// `casscf` only: return canonical core and virtual and natural active
    /// orbitals. Default `false`.
    pub canonicalize: Option<bool>,
    /// Orbital-gradient norm below which CASSCF is converged. Default `1e-7`.
    pub grad_tol: Option<f64>,
    /// Energy change (Ha) that must also be met at the same iteration.
    /// Default `1e-12`.
    pub energy_tol: Option<f64>,
    /// CASSCF macro iterations. Default `300`.
    pub max_iter: Option<usize>,
    /// Cap on the largest orbital-rotation step element (rad). Default `0.2`.
    pub max_step: Option<f64>,
    /// Orbital step: `"ah"` (default; augmented Hessian on the exact orbital
    /// Hessian at fixed RDMs, with a trust radius: fewer macro iterations,
    /// each with Davidson Hessian-vector products) or `"bfgs"` (inverse BFGS
    /// from an approximate diagonal Hessian).
    pub optimizer: Option<String>,
    /// Two-electron integrals of the CASSCF iterations: `"auto"` (default;
    /// dense when the block fits in half the memory budget), `"direct"`
    /// (direct J/K and `(pu|vw)` builds, no `nbas^4` array), `"dense"`
    /// (the `nbas^4` AO block held in memory) or `"ri"` (density-fitted
    /// integrals in `auxbasis`, `naux nbas^2` held; an approximation of the
    /// Hamiltonian, also used by `nevpt2`).
    pub integrals: Option<String>,
    /// `integrals = "direct"` only: Schwarz screening threshold. Default
    /// `1e-12`.
    pub integral_thresh: Option<f64>,
    /// `integrals = "ri"` only, and required there: the bundled auxiliary
    /// basis (e.g. `"def2-universal-jkfit"`). No default.
    pub auxbasis: Option<String>,
    /// Second-order check of the converged CASSCF point: `"off"` (default),
    /// `"check"` (the lowest eigenvalue of the CI-relaxed orbital Hessian is
    /// printed and recorded on the certificate, and a saddle is capped to not
    /// graded) or `"follow"` (as `"check"`, and at a saddle the orbitals step
    /// along the negative mode and CASSCF converges again).
    pub stability: Option<String>,
    /// `[cas.ci]`: the CAS-CI eigensolver.
    pub ci: Option<CasCiCfg>,
}

/// `[cas.ci]`: the CAS-CI Davidson of every CASSCF macro iteration (or of
/// the CAS-CI reference). Python `ci_conv_thresh`, `ci_max_subspace`,
/// `ci_max_iter`.
#[derive(Deserialize, Debug, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct CasCiCfg {
    /// Residual `‖Hc − Ec‖` below which a root is accepted. Default `1e-10`.
    pub conv_thresh: Option<f64>,
    /// Davidson subspace size before a restart. Default `50`.
    pub max_subspace: Option<usize>,
    /// Davidson iterations. Default `200`.
    pub max_iter: Option<usize>,
}

/// The validated CAS run of a config ([`resolve_plan`]).
#[derive(Debug, Clone)]
pub struct CasPlan {
    /// `None` for `casscf`, the SC-NEVPT2 reference for `nevpt2`.
    pub nevpt2: Option<Nevpt2On>,
    /// `[cas] n_core` as written.
    pub n_core: Option<usize>,
    /// The CASSCF / CAS-CI settings; `n_core` is filled in at run time.
    pub cas: CasScfConfig,
    /// The written `[scf]` convergence keys.
    pub scf: ScfStart,
}

/// The sections a CAS run reads.
pub(crate) const READ_SECTIONS: &[&str] = &[
    "molecule", "basis", "method", "cas", "scf", "memory", "parallel", "output",
];

/// The `[scf]` keys a CAS run reads.
pub(crate) const READ_SCF_KEYS: &[&str] = &["max_iter", "energy_conv", "density_conv"];

fn positive(name: &str, v: Option<f64>) -> Result<(), String> {
    match v {
        Some(x) if !(x.is_finite() && x > 0.0) => Err(format!(
            "[cas] {name} must be a positive finite number, got {x}"
        )),
        _ => Ok(()),
    }
}

fn at_least_one(name: &str, v: Option<usize>) -> Result<(), String> {
    match v {
        Some(0) => Err(format!("[cas] {name} must be at least 1")),
        _ => Ok(()),
    }
}

/// The keys written in the raw TOML table `table` (empty if absent).
fn written(table: Option<&toml::Value>) -> Vec<&str> {
    table
        .and_then(|v| v.as_table())
        .map(|t| t.keys().map(String::as_str).collect())
        .unwrap_or_default()
}

/// Every section and `[scf]` key outside what a CAS run reads is an error.
fn check_sections(raw: &toml::Value, kind: &str) -> Result<(), String> {
    let Some(top) = raw.as_table() else {
        return Ok(());
    };
    for section in top.keys() {
        if !READ_SECTIONS.contains(&section.as_str()) {
            return Err(format!(
                "[{section}] is not read by method.kind = \"{kind}\", which reads [cas], \
                 [cas.ci], [scf] {}, [memory], [parallel] and [output]; remove it",
                READ_SCF_KEYS.join(" / ")
            ));
        }
    }
    for key in written(raw.get("scf")) {
        if !READ_SCF_KEYS.contains(&key) {
            return Err(format!(
                "[scf] {key} is not read by method.kind = \"{kind}\": its RHF start is the \
                 exact-J/K Coulomb RHF with the library defaults, as in Python run_casscf / \
                 run_nevpt2, and reads only [scf] {}",
                READ_SCF_KEYS.join(", ")
            ));
        }
    }
    Ok(())
}

/// The keys of `[cas]` that `kind` (with `on` for `nevpt2`) does not read.
fn check_cas_keys(cas: &CasCfg, kind: &str, on: Option<Nevpt2On>) -> Result<(), String> {
    if kind == "casscf" && cas.reference.is_some() {
        return Err(
            "[cas] reference is read by method.kind = \"nevpt2\" only (the SC-NEVPT2 \
                    reference); remove it"
                .into(),
        );
    }
    if kind == "nevpt2" && cas.canonicalize.is_some() {
        return Err(
            "[cas] canonicalize is read by method.kind = \"casscf\" only; SC-NEVPT2 \
                    canonicalizes its core and virtual orbitals itself"
                .into(),
        );
    }
    if on == Some(Nevpt2On::Casci) {
        let orbital_keys = [
            ("grad_tol", cas.grad_tol.is_some()),
            ("energy_tol", cas.energy_tol.is_some()),
            ("max_iter", cas.max_iter.is_some()),
            ("max_step", cas.max_step.is_some()),
            ("optimizer", cas.optimizer.is_some()),
            ("integrals", cas.integrals.is_some()),
            ("integral_thresh", cas.integral_thresh.is_some()),
            ("auxbasis", cas.auxbasis.is_some()),
            ("stability", cas.stability.is_some()),
        ];
        if let Some((k, _)) = orbital_keys.iter().find(|(_, set)| *set) {
            return Err(format!(
                "[cas] {k} sets the CASSCF orbital optimization, which reference = \"casci\" \
                 does not run; remove it"
            ));
        }
    }
    if on.is_some() && cas.n_elec_active[0] != cas.n_elec_active[1] {
        return Err(format!(
            "[cas] n_elec_active = {:?}: SC-NEVPT2 is implemented for closed-shell active \
             spaces (n_alpha = n_beta) only; an open-shell sector runs with kind = \"casscf\"",
            cas.n_elec_active
        ));
    }
    Ok(())
}

/// The CAS plan of `cfg` (whose raw TOML text is `text`): `None` for a kind
/// other than `casscf` / `nevpt2`, which must then have no `[cas]` section.
pub fn resolve_plan(cfg: &Config, text: &str) -> Result<Option<CasPlan>, String> {
    let kind = cfg.method.kind.as_str();
    if !matches!(kind, "casscf" | "nevpt2") {
        return match cfg.cas {
            Some(_) => Err(format!(
                "[cas] is read by method.kind = \"casscf\" and \"nevpt2\" only; kind = \
                 \"{kind}\" would ignore it"
            )),
            None => Ok(None),
        };
    }
    if cfg.method.task != "energy" {
        return Err(format!(
            "method.kind = \"{kind}\" supports task = \"energy\" only (got \"{}\"): there is \
             no CASSCF or SC-NEVPT2 nuclear gradient",
            cfg.method.task
        ));
    }
    let Some(cas) = cfg.cas.as_ref() else {
        return Err(format!(
            "method.kind = \"{kind}\" needs a [cas] section with n_active and n_elec_active"
        ));
    };
    if cfg.molecule.multiplicity != 1 {
        return Err(format!(
            "method.kind = \"{kind}\" starts from a closed-shell RHF, so [molecule] \
             multiplicity must be 1 (got {}); the spin of the CAS state is set by [cas] \
             n_elec_active = [n_alpha, n_beta], e.g. [5, 3] for the O2 triplet in CAS(8,6)",
            cfg.molecule.multiplicity
        ));
    }
    let raw: toml::Value = toml::from_str(text).map_err(|e| e.to_string())?;
    check_sections(&raw, kind)?;
    let on = match kind {
        "nevpt2" => Some(
            Nevpt2On::parse_config_str(cas.reference.as_deref().unwrap_or("casscf"))
                .map_err(|e| format!("[cas] {e}"))?,
        ),
        _ => None,
    };
    check_cas_keys(cas, kind, on)?;
    let ci = cas.ci.clone().unwrap_or_default();
    positive("grad_tol", cas.grad_tol)?;
    positive("energy_tol", cas.energy_tol)?;
    positive("max_step", cas.max_step)?;
    positive("ci.conv_thresh", ci.conv_thresh)?;
    at_least_one("max_iter", cas.max_iter)?;
    at_least_one("ci.max_iter", ci.max_iter)?;
    at_least_one("ci.max_subspace", ci.max_subspace)?;
    let d = CasScfConfig::default();
    let integrals = match (
        cas.integrals.as_deref(),
        cas.integral_thresh,
        cas.auxbasis.as_deref(),
    ) {
        (None, None, None) => d.integrals,
        (route, t, aux) => {
            let default_route = if t.is_some() { "direct" } else { "auto" };
            CasScfIntegrals::parse_config_str(route.unwrap_or(default_route), t, aux)
                .map_err(|e| format!("[cas] {e}"))?
        }
    };
    let optimizer = match cas.optimizer.as_deref() {
        None => d.optimizer,
        Some(o) => CasScfOptimizer::parse_config_str(o).map_err(|e| format!("[cas] {e}"))?,
    };
    let stability = match cas.stability.as_deref() {
        None => d.stability,
        Some(m) => CasScfStabilityMode::parse_config_str(m).map_err(|e| format!("[cas] {e}"))?,
    };
    let scf_keys = written(raw.get("scf"));
    let scf_key = |k: &str| scf_keys.contains(&k);
    Ok(Some(CasPlan {
        nevpt2: on,
        n_core: cas.n_core,
        cas: CasScfConfig {
            n_active: cas.n_active,
            n_elec_active: (cas.n_elec_active[0], cas.n_elec_active[1]),
            n_core: 0,
            grad_tol: cas.grad_tol.unwrap_or(d.grad_tol),
            energy_tol: cas.energy_tol.unwrap_or(d.energy_tol),
            max_iter: cas.max_iter.unwrap_or(d.max_iter),
            max_step: cas.max_step.unwrap_or(d.max_step),
            ci_conv_thresh: ci.conv_thresh.unwrap_or(d.ci_conv_thresh),
            ci_max_subspace: ci.max_subspace.unwrap_or(d.ci_max_subspace),
            ci_max_iter: ci.max_iter.unwrap_or(d.ci_max_iter),
            memory_budget_bytes: cfg.memory.budget_bytes(),
            canonicalize: cas.canonicalize.unwrap_or(false),
            integrals,
            optimizer,
            stability,
            ..d
        },
        scf: ScfStart {
            max_iter: scf_key("max_iter").then_some(cfg.scf.max_iter),
            energy_conv: scf_key("energy_conv").then_some(cfg.scf.energy_conv),
            density_conv: scf_key("density_conv").then_some(cfg.scf.density_conv),
        },
    }))
}

fn build_stamp() -> BuildStamp {
    BuildStamp::new(
        ferric_build_info::VERSION,
        ferric_build_info::COMMIT,
        ferric_build_info::DIRTY,
    )
}

fn classes_json(c: &Nevpt2Classes) -> serde_json::Value {
    let m: serde_json::Map<String, serde_json::Value> = Nevpt2Classes::NAMES
        .iter()
        .zip(c.to_array())
        .map(|(n, v)| (n.to_string(), serde_json::json!(v)))
        .collect();
    serde_json::Value::Object(m)
}

/// The run-log record of a CASSCF second-order check (`null` without one).
fn stability_json(st: Option<&ferric_ci::CasScfStability>) -> serde_json::Value {
    match st {
        None => serde_json::Value::Null,
        Some(s) => serde_json::json!({
            "lowest_eigenvalue": s.lowest_eigenvalue,
            "verdict": s.verdict().label(),
            "noise_floor": s.noise_floor,
            "residual": s.residual,
            "products": s.products,
            "follow_steps": s.follow_steps,
            "energy_before_follow": s.energy_before_follow,
        }),
    }
}

/// Run `cfg`'s CAS plan; `gpu_flag` is whether `--gpu` was given (refused:
/// the CAS kinds run on the CPU).
pub fn run_cas(cfg: &Config, gpu_flag: bool) {
    let Some(plan) = cfg.cas_plan.as_ref() else {
        die("internal: run_cas called without a CAS plan");
    };
    let kind = cfg.method.kind.as_str();
    if gpu_flag {
        die(format!(
            "--gpu is not read by method.kind = \"{kind}\", which runs on the CPU"
        ));
    }
    let mol = Molecule::load_xyz_with_charge(
        &cfg.molecule.xyz,
        cfg.molecule.charge,
        cfg.molecule.multiplicity,
    )
    .unwrap_or_else(|e| die(e));
    let bs = match (&cfg.basis.name, &cfg.basis.path) {
        (Some(name), _) => basis::bundled(name),
        (None, Some(path)) => basis::load_g94(path),
        (None, None) => Err(ferric_core::FerricError::Basis("no basis specified".into())),
    }
    .unwrap_or_else(|e| die(e));
    let ecp = EcpMolecule::new(mol, &bs);
    let mut cas = plan.cas;
    cas.n_core = match plan.n_core {
        Some(n) => n,
        None => {
            let nelec = ecp.mol().nelec() as usize;
            ferric_ci::driver::default_n_core(nelec, cas.n_elec_active, "[cas]")
                .unwrap_or_else(|e| die(e))
        }
    };
    let resolution = ferric_core::memory::resolve_budget(cas.memory_budget_bytes);
    ferric_core::memory::pool::install_global(
        ferric_core::memory::pool::MemoryPool::with_capacity_bytes(resolution.bytes),
    );
    eprintln!("[ferric] {}", resolution.audit_line());
    let sys = ecp
        .prepare(PrepareOptions::default())
        .unwrap_or_else(|e| die(e));
    let (mol, prep) = (sys.mol(), sys.basis());
    if let Some(rl) = ferric_scf::runlog::log() {
        ferric_scf::runlog::set_build_identity(
            ferric_build_info::VERSION,
            ferric_build_info::short_commit().as_deref(),
        );
        rl.run_start(
            serde_json::json!({
                "method": kind,
                "task": cfg.method.task,
                "basis": bs.name,
                "n_active": cas.n_active,
                "n_elec_active": [cas.n_elec_active.0, cas.n_elec_active.1],
                "n_core": cas.n_core,
                "reference": plan.nevpt2.map(Nevpt2On::as_str),
                "max_iter": plan.scf.max_iter,
                "energy_conv": plan.scf.energy_conv,
                "density_conv": plan.scf.density_conv,
                "memory_budget_bytes": resolution.bytes,
                "openblas_num_threads": std::env::var("OPENBLAS_NUM_THREADS").ok(),
                "rayon_num_threads": rayon::current_num_threads(),
            }),
            serde_json::json!({
                "path": cfg.molecule.xyz,
                "n_atoms": mol.atoms.len(),
                "charge": cfg.molecule.charge,
                "multiplicity": cfg.molecule.multiplicity,
                "n_electrons": mol.nelec(),
                "n_basis": prep.nbasis(),
            }),
        );
    }
    let rhf = rhf_start(&sys, plan.scf, cas.memory_budget_bytes).unwrap_or_else(|e| die(e));
    let (na, nb) = cas.n_elec_active;
    let header = |name: &str| {
        println!("{name}/{} on {}", bs.name, cfg.molecule.xyz);
        println!(
            "  active space  = CAS({}, {}): ({na}, {nb}) electrons above {} core orbitals",
            na + nb,
            cas.n_active,
            cas.n_core
        );
        println!("  nbasis        = {}", prep.nbasis());
        println!("  RHF energy    = {:.10} Hartree", rhf.energy);
    };
    let (energy, converged) = match plan.nevpt2 {
        None => {
            let r = casscf_on_rhf(&sys, &rhf, &cas).unwrap_or_else(|e| die(e));
            let cert = ferric_mp2::certificate::with_anchor_envelope(
                ferric_ci::certificate::casscf_certificate(&r, &cas).with_build(build_stamp()),
                "CASSCF",
                mol,
                &bs,
                None,
            );
            header("CASSCF");
            println!("  determinants  = {}", r.n_determinants);
            println!(
                "  iterations    = {} (|g| = {:.2e}, converged: {})",
                r.iterations, r.grad_norm, r.converged
            );
            println!("  CASSCF energy = {:.10} Hartree", r.e_total);
            if let Some(st) = &r.stability {
                println!("  stability     = {}", st.summary());
            }
            if !r.converged {
                eprintln!(
                    "warning: {}",
                    ferric_ci::certificate::casscf_unconverged_warning(&r, &cas)
                );
            }
            let cert_json = crate::print_certificate(&cert);
            if let Some(rl) = ferric_scf::runlog::log() {
                rl.result(
                    "casscf",
                    r.e_total,
                    serde_json::json!({
                        "e_core": r.e_core,
                        "e_scf_reference": rhf.energy,
                        "n_determinants": r.n_determinants,
                        "iterations": r.iterations,
                        "grad_norm": r.grad_norm,
                        "converged": r.converged,
                        "ci_residual_norm": r.ci_residual_norm,
                        "active_occupations": r.rdms.dm1.diag().to_vec(),
                        "mo_energy": r.mo_energy,
                        "stability": stability_json(r.stability.as_ref()),
                        "certificate": cert_json,
                    }),
                );
            }
            (r.e_total, r.converged)
        }
        Some(on) => {
            let run = nevpt2_on_rhf(&sys, &rhf, on, &cas).unwrap_or_else(|e| die(e));
            let cert = ferric_mp2::certificate::with_anchor_envelope(
                ferric_ci::certificate::nevpt2_certificate(
                    on,
                    &cas,
                    run.reference_stability.as_ref(),
                )
                .with_build(build_stamp()),
                "SC-NEVPT2",
                mol,
                &bs,
                None,
            );
            let r = &run.result;
            let total = r.e_reference + r.e_corr;
            header("SC-NEVPT2");
            println!("  reference     = {}", on.as_str());
            println!("  E(reference)  = {:.10} Hartree", r.e_reference);
            if let Some(st) = &run.reference_stability {
                println!("  stability     = {}", st.summary());
            }
            for (name, v) in Nevpt2Classes::NAMES.iter().zip(r.energies.to_array()) {
                println!("  {name:<13} = {v:.10} Hartree");
            }
            println!("  NEVPT2 corr   = {:.10} Hartree", r.e_corr);
            println!("  Total         = {total:.10} Hartree");
            let cert_json = crate::print_certificate(&cert);
            if let Some(rl) = ferric_scf::runlog::log() {
                rl.result(
                    "nevpt2",
                    total,
                    serde_json::json!({
                        "e_corr": r.e_corr,
                        "e_reference": r.e_reference,
                        "reference": on.as_str(),
                        "e_scf_reference": rhf.energy,
                        "classes": classes_json(&r.energies),
                        "norms": classes_json(&r.norms),
                        "reference_iterations": run.reference_iterations,
                        "reference_grad_norm": run.reference_grad_norm,
                        "reference_stability": stability_json(run.reference_stability.as_ref()),
                        "ci_residual_norm": r.ci_residual_norm,
                        "n_core": r.n_core,
                        "n_active": r.n_active,
                        "n_virtual": r.n_virtual,
                        "certificate": cert_json,
                    }),
                );
            }
            (total, true)
        }
    };
    if let Some(rl) = ferric_scf::runlog::log() {
        rl.run_end(
            energy,
            converged,
            if converged {
                "converged"
            } else {
                "not_converged"
            },
            serde_json::json!({
                "method": kind,
                "task": cfg.method.task,
                "energy_is": "total",
                "scf_iterations": rhf.iterations,
            }),
        );
    }
}
