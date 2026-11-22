//! `method.kind = "opmp3"`: orbital-partitioned Padé MP3 (OP-MP3, SOP-MP3,
//! SSD-OPMP3) through `ferric_mp2::opmp3_driver`, the code the Python
//! `run_opmp3` runs.
//!
//! The run is its own restricted reference SCF (RHF/ROHF, or RKS/ROKS for a
//! functional; exact J/K), then the HF-Fock semicanonical reference with
//! non-Brillouin singles, so the CLI reads only `[opmp3]`, `[local]` (the
//! amplitude-threshold local OP-MP3, `ferric_mp2::local_opmp3`; RI and
//! closed shell), `[scf]`
//! `max_iter` / `energy_conv` / `density_conv`, `[memory]`, `[parallel]` and
//! `[output]`. [`resolve_plan`] refuses every other section and key at load
//! time.

use crate::config::Config;
use crate::error::die;
use ferric_core::basis;
use ferric_core::certificate::BuildStamp;
use ferric_core::mol::Molecule;
use ferric_mp2::lmp2_amplitude::AmplitudeLmp2Config;
use ferric_mp2::local_opmp3::{local_opmp3_on, LocalOpmp3Result};
use ferric_mp2::opmp3_driver::{
    opmp3_on, opmp3_scf, opmp3_scf_config, Opmp3Config, Opmp3Integrals, Opmp3RefKind, Opmp3Result,
};
use ferric_scf::PreparedSystem;
use serde::Deserialize;

/// `[opmp3]`: the reference and model settings of `method.kind = "opmp3"`.
/// Python `run_opmp3` takes the same settings (`reference`, `x_ss`,
/// `auxbasis`, `frozen_core`).
#[derive(Deserialize, Debug, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct Opmp3Cfg {
    /// `"hf"` (default; RHF or ROHF) or a functional name (RKS or ROKS).
    pub reference: Option<String>,
    /// SSD-OPMP3 same-spin damping. Default: the paper's printed value for the
    /// reference (hf 0, pbe0 0.5, pbe 0.7, spw92 0.75, r2scan 0.6, wb97x-v
    /// 0.5, wb97m-v 0.5); a reference without one needs it written.
    pub x_ss: Option<f64>,
    /// Amplitude-stage integrals: `"exact"` (default, the paper's) or `"ri"`.
    pub integrals: Option<String>,
    /// `integrals = "ri"` only, and required there: the bundled auxiliary
    /// basis. No default.
    pub auxbasis: Option<String>,
    /// Lowest occupied orbitals of each spin left uncorrelated. Default 0
    /// (the paper correlates every electron).
    pub frozen_core: Option<usize>,
}

/// The validated OP-MP3 run of a config ([`resolve_plan`]).
#[derive(Debug, Clone)]
pub struct Opmp3Plan {
    /// The library settings.
    pub config: Opmp3Config,
    /// The RI auxiliary basis name, `None` for exact integrals.
    pub auxbasis: Option<String>,
    /// `[local] scheme = "amplitude-threshold"`: the pattern config, and
    /// whether to also run the canonical OP-MP3 (`[local] reference`).
    pub local: Option<(AmplitudeLmp2Config, bool)>,
}

/// The sections an OP-MP3 run reads.
pub(crate) const READ_SECTIONS: &[&str] = &[
    "molecule", "basis", "method", "opmp3", "local", "scf", "memory", "parallel", "output",
];

/// The `[scf]` keys an OP-MP3 run reads.
pub(crate) const READ_SCF_KEYS: &[&str] = &["max_iter", "energy_conv", "density_conv"];

/// The keys written in the raw TOML table `name` (empty if absent).
fn written<'a>(raw: &'a toml::Value, name: &str) -> Vec<&'a str> {
    raw.get(name)
        .and_then(|v| v.as_table())
        .map(|t| t.keys().map(String::as_str).collect())
        .unwrap_or_default()
}

/// Every section and `[scf]` key outside what an OP-MP3 run reads is an error.
fn check_sections(raw: &toml::Value) -> Result<(), String> {
    if let Some(top) = raw.as_table() {
        if let Some(s) = top.keys().find(|s| !READ_SECTIONS.contains(&s.as_str())) {
            return Err(format!(
                "[{s}] is not read by method.kind = \"opmp3\", which reads [opmp3], [local], \
                 [scf] {}, [memory], [parallel] and [output]; remove it",
                READ_SCF_KEYS.join(" / ")
            ));
        }
    }
    if let Some(k) = written(raw, "scf")
        .into_iter()
        .find(|k| !READ_SCF_KEYS.contains(k))
    {
        return Err(format!(
            "[scf] {k} is not read by method.kind = \"opmp3\": its reference is the exact-J/K \
             RHF/ROHF (RKS/ROKS with [opmp3] reference) of Python run_opmp3, which reads only \
             [scf] {}",
            READ_SCF_KEYS.join(", ")
        ));
    }
    Ok(())
}

/// The RI auxiliary basis of `[opmp3]`, `None` for exact integrals.
fn auxbasis_of(o: &Opmp3Cfg) -> Result<Option<String>, String> {
    match (o.integrals.as_deref().unwrap_or("exact"), &o.auxbasis) {
        ("exact", None) => Ok(None),
        ("exact", Some(_)) => Err(
            "[opmp3] auxbasis is read with integrals = \"ri\" only; remove it or set \
             integrals = \"ri\""
                .into(),
        ),
        ("ri", Some(a)) => Ok(Some(a.clone())),
        ("ri", None) => Err("[opmp3] integrals = \"ri\" needs auxbasis (no default)".into()),
        (other, _) => Err(format!(
            "[opmp3] integrals = {other:?}: expected \"exact\" or \"ri\""
        )),
    }
}

/// The OP-MP3 plan of `cfg` (whose raw TOML text is `text`): `None` for a
/// kind other than `opmp3`, which must then have no `[opmp3]` section.
pub fn resolve_plan(cfg: &Config, text: &str) -> Result<Option<Opmp3Plan>, String> {
    if cfg.method.kind != "opmp3" {
        return match cfg.opmp3 {
            Some(_) => Err(format!(
                "[opmp3] is read by method.kind = \"opmp3\" only; kind = \"{}\" would ignore it",
                cfg.method.kind
            )),
            None => Ok(None),
        };
    }
    if cfg.method.task != "energy" {
        return Err(format!(
            "method.kind = \"opmp3\" supports task = \"energy\" only (got \"{}\"): there is no \
             OP-MP3 nuclear gradient",
            cfg.method.task
        ));
    }
    let raw: toml::Value = toml::from_str(text).map_err(|e| e.to_string())?;
    check_sections(&raw)?;
    let o = cfg.opmp3.clone().unwrap_or_default();
    let reference = Opmp3RefKind::parse_config_str(o.reference.as_deref().unwrap_or("hf"))
        .map_err(|e| format!("[opmp3] {e}"))?;
    let auxbasis = auxbasis_of(&o)?;
    let scf_keys = written(&raw, "scf");
    let scf_key = |k: &str| scf_keys.contains(&k);
    let config = Opmp3Config {
        x_ss: o.x_ss,
        frozen_core: o.frozen_core.unwrap_or(0),
        scf_max_iter: scf_key("max_iter").then_some(cfg.scf.max_iter),
        scf_energy_conv: scf_key("energy_conv").then_some(cfg.scf.energy_conv),
        scf_density_conv: scf_key("density_conv").then_some(cfg.scf.density_conv),
        memory_budget_bytes: cfg.memory.budget_bytes(),
        ..Opmp3Config::new(reference)
    };
    config.resolved_x_ss().map_err(|e| format!("[opmp3] {e}"))?;
    // the shared [local] rules (eps, integral_direct and the map keys per kind)
    cfg.validate_local()?;
    let local = match cfg.local_model()? {
        None => None,
        Some(model) => {
            if auxbasis.is_none() {
                return Err(
                    "[local] scheme = \"amplitude-threshold\" with method.kind = \
                            \"opmp3\" runs on RI integrals: set [opmp3] integrals = \"ri\" and \
                            auxbasis"
                        .into(),
                );
            }
            let lcfg = model.amplitude_config(config.frozen_core, config.memory_budget_bytes);
            Some((lcfg, model.reference))
        }
    };
    Ok(Some(Opmp3Plan {
        config,
        auxbasis,
        local,
    }))
}

/// Run `cfg`'s OP-MP3 plan; `gpu_flag` is whether `--gpu` was given (refused:
/// the kind runs on the CPU).
pub fn run_opmp3(cfg: &Config, gpu_flag: bool) {
    let Some(plan) = cfg.opmp3_plan.as_ref() else {
        die("internal: run_opmp3 called without an OP-MP3 plan");
    };
    if gpu_flag {
        die("--gpu is not read by method.kind = \"opmp3\", which runs on the CPU");
    }
    let config = &plan.config;
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
    let resolution = ferric_core::memory::resolve_budget(config.memory_budget_bytes);
    ferric_core::memory::pool::install_global(
        ferric_core::memory::pool::MemoryPool::with_capacity_bytes(resolution.bytes),
    );
    eprintln!("[ferric] {}", resolution.audit_line());
    let sys = PreparedSystem::new(mol, &bs).unwrap_or_else(|e| die(e));
    let (mol, prep) = (sys.mol(), sys.basis());
    let aux = plan.auxbasis.as_deref().map(|name| {
        let a = basis::bundled(name).unwrap_or_else(|e| die(e));
        sys.prepare_aux(&a).unwrap_or_else(|e| die(e))
    });
    let integrals = aux
        .as_ref()
        .map_or(Opmp3Integrals::Exact, Opmp3Integrals::Ri);
    let x_ss = config.resolved_x_ss().unwrap_or_else(|e| die(e));
    if let Some(rl) = ferric_scf::runlog::log() {
        ferric_scf::runlog::set_build_identity(
            ferric_build_info::VERSION,
            ferric_build_info::short_commit().as_deref(),
        );
        rl.run_start(
            serde_json::json!({
                "method": "opmp3",
                "task": cfg.method.task,
                "basis": bs.name,
                "reference": config.reference.name(),
                "x_ss": x_ss,
                "auxbasis": plan.auxbasis,
                "frozen_core": config.frozen_core,
                "memory_budget_bytes": resolution.bytes,
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
    let scf = opmp3_scf(&sys, config).unwrap_or_else(|e| die(e));
    let canonical = || opmp3_on(&sys, &scf, integrals, config).unwrap_or_else(|e| die(e));
    let (r, local) = match (&plan.local, aux.as_ref()) {
        (Some((lcfg, with_reference)), Some(aux)) => {
            let lr = local_opmp3_on(aux, &scf, config, lcfg).unwrap_or_else(|e| die(e));
            let reference = with_reference.then(canonical);
            (lr.opmp3.clone(), Some((lr, reference)))
        }
        (Some(_), None) => die("internal: local OP-MP3 plan without an auxiliary basis"),
        (None, _) => (canonical(), None),
    };
    let scf_config = opmp3_scf_config(config);
    let reference = ferric_scf::certificate::with_scf_envelope(
        crate::scf_certificate_stamped(mol, &scf, &scf_config, None),
        mol,
        &scf,
        &scf_config,
        &bs,
    );
    let cert = match (&local, plan.auxbasis.as_deref()) {
        (Some((lr, _)), Some(a)) => ferric_mp2::certificate::local_opmp3_certificate(
            &reference,
            a,
            config.frozen_core,
            x_ss,
            lr.eps,
            lr.keep_fraction,
        ),
        _ => ferric_mp2::certificate::opmp3_certificate(
            &reference,
            plan.auxbasis.as_deref(),
            config.frozen_core,
            x_ss,
        ),
    }
    .with_build(BuildStamp::new(
        ferric_build_info::VERSION,
        ferric_build_info::COMMIT,
        ferric_build_info::DIRTY,
    ));
    let m = &r.models;
    let models = [
        ("MP2", m.mp2),
        ("MP3", m.mp3),
        ("P-MP3", m.p_mp3),
        ("OP-MP3", m.op_mp3),
        ("SOP-MP3", m.sop_mp3),
        ("SSD-OPMP3", m.ssd_opmp3),
    ];
    println!("OP-MP3/{} on {}", bs.name, cfg.molecule.xyz);
    println!("  reference     = {}", config.reference.name());
    println!("  nbasis        = {}", prep.nbasis());
    println!("  E(SCF)        = {:.10} Hartree", r.e_scf);
    println!("  E_HF[rho]     = {:.10} Hartree", r.e_hf);
    println!("  E_NBS         = {:.10} Hartree", r.e_nbs);
    println!("  x_ss          = {x_ss}");
    for (name, e) in models {
        println!(
            "  {name:<13} = {:.10} Hartree (correlation {e:.10})",
            r.total(e)
        );
    }
    if let Some((lr, reference)) = &local {
        print_local(lr, reference.as_ref());
    }
    let cert_json = crate::print_certificate(&cert);
    let total = r.total(m.ssd_opmp3);
    if let Some(rl) = ferric_scf::runlog::log() {
        let corr: serde_json::Map<String, serde_json::Value> = models
            .iter()
            .map(|(k, e)| (k.to_string(), serde_json::json!(e)))
            .collect();
        rl.result(
            "opmp3",
            total,
            serde_json::json!({
                "e_scf": r.e_scf,
                "e_hf": r.e_hf,
                "e_nbs": r.e_nbs,
                "x_ss": x_ss,
                "correlation": corr,
                "max_ov": r.max_ov,
                "scf_iterations": r.scf_iterations,
                "local": local.as_ref().map(|(lr, reference)| local_json(lr, reference.as_ref())),
                "certificate": cert_json,
            }),
        );
        rl.run_end(
            total,
            true,
            "converged",
            serde_json::json!({
                "method": "opmp3",
                "task": cfg.method.task,
                "energy_is": "total (SSD-OPMP3)",
                "scf_iterations": r.scf_iterations,
            }),
        );
    }
}

/// The local block of the printout: the model (ε, kept fraction), the
/// solves, the one-pattern E3 identity, and with `[local] reference` the
/// error of every model against the canonical OP-MP3.
fn print_local(lr: &LocalOpmp3Result, reference: Option<&Opmp3Result>) {
    println!(
        "  local         = amplitude-threshold eps = {:e} (kept {:.6} of the amplitudes, {} pairs)",
        lr.eps, lr.keep_fraction, lr.n_pairs
    );
    println!(
        "  local maps    = lambda_boys {}, virtual space {:?}",
        lr.lambda_boys.lambda, lr.virtual_space
    );
    println!(
        "  local solves  = c1 {} iters (relres {:.1e}), c2 {} iters (relres {:.1e}); \
         |E3 - E3(2n+1)| = {:.2e}",
        lr.cg_iterations[0],
        lr.cg_relres[0],
        lr.cg_iterations[1],
        lr.cg_relres[1],
        lr.e3_identity_gap()
    );
    if let Some(c) = reference {
        let (l, m) = (&lr.opmp3.models, &c.models);
        for (name, a, b) in [
            ("MP2", l.mp2, m.mp2),
            ("OP-MP3", l.op_mp3, m.op_mp3),
            ("SSD-OPMP3", l.ssd_opmp3, m.ssd_opmp3),
        ] {
            println!(
                "  local error   {name:<10} = {:+.3e} Hartree vs canonical",
                a - b
            );
        }
    }
}

/// The run-log record of a local OP-MP3 run.
fn local_json(lr: &LocalOpmp3Result, reference: Option<&Opmp3Result>) -> serde_json::Value {
    serde_json::json!({
        "scheme": "amplitude-threshold",
        "eps": lr.eps,
        "keep_fraction": lr.keep_fraction,
        "n_pairs": lr.n_pairs,
        "lambda_boys": lr.lambda_boys.lambda,
        "cg_iterations": lr.cg_iterations,
        "cg_relres": lr.cg_relres,
        "e3_identity_gap": lr.e3_identity_gap(),
        "canonical_correlation": reference.map(|c| serde_json::json!({
            "mp2": c.models.mp2,
            "op-mp3": c.models.op_mp3,
            "ssd-opmp3": c.models.ssd_opmp3,
        })),
    })
}
