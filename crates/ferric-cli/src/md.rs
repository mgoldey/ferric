//! `task = "md"` driver for the CLI: Born-Oppenheimer molecular dynamics on
//! the SCF surface of `method.kind` (`ferric_scf::run_aimd`, the driver the
//! Python `run_aimd` calls), configured by `[md]`.
//!
//! Every recorded frame is printed as a table row, appended to a multi-frame
//! XYZ trajectory (Angstrom) and to an energy CSV (`[md] trajectory` /
//! `energies`, default `<input stem>.md.xyz` / `.md.csv`). The run ends with a
//! summary, an ungraded `md:<kind>` certificate whose reference is the SCF at
//! the final geometry, and an `md` run-log record.

use crate::config::{Config, MdCfg};
use ferric_core::basis::BasisSet;
use ferric_core::certificate::Certificate;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::operator::Operator;
use ferric_md::{
    EnergyCsvWriter, MdObserver, MdResult, MdState, MdSystem, Thermostat, XyzTrajectoryWriter,
};
use ferric_scf::rhf::RhfConfig;
use std::path::Path;

fn die(what: &str, e: impl std::fmt::Display) -> ! {
    eprintln!("error{what}: {e}");
    std::process::exit(1);
}

/// Prints one table row per recorded frame.
struct FramePrinter;

impl MdObserver for FramePrinter {
    fn on_frame(&mut self, _system: &MdSystem, state: &MdState) -> Result<(), FerricError> {
        let f = &state.frame;
        println!(
            "  {:>6} {:>10.3} {:>18.10} {:>14.10} {:>18.10} {:>10.2}",
            f.step, f.time_fs, f.potential, f.kinetic, f.conserved, f.temperature_k
        );
        Ok(())
    }
}

/// The SCF reference of `method` (after `Config::dispatch_kind`).
fn md_spin(method: &str) -> ferric_scf::Spin {
    match method {
        "uhf" => ferric_scf::Spin::Unrestricted,
        "rohf" => ferric_scf::Spin::RestrictedOpen,
        _ => ferric_scf::Spin::Restricted,
    }
}

/// `method.task = "md"`. `load_config` has already checked `[md]` and the
/// kind (`Config::validate_md`); every failure here exits 1.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_md(
    method: &str,
    cfg: &Config,
    ctx: &ParallelContext,
    mol: &Molecule,
    bs: &BasisSet,
    op: Operator,
    rhf_config: &RhfConfig,
    toml_path: &Path,
) {
    let md_cfg: &MdCfg = cfg
        .md
        .as_ref()
        .unwrap_or_else(|| die("", "method.task = \"md\" needs an [md] section"));
    let md = md_cfg.md_config().unwrap_or_else(|e| die("", e));
    let spin = md_spin(method);
    let model = crate::dispersion_gradient_model(cfg, ctx, mol, bs, op, rhf_config, "md", spin);
    let (xyz_path, csv_path) = md_cfg.output_paths(toml_path);
    let mut xyz =
        XyzTrajectoryWriter::create(&xyz_path).unwrap_or_else(|e| die(" in [md] trajectory", e));
    let mut csv =
        EnergyCsvWriter::create(&csv_path).unwrap_or_else(|e| die(" in [md] energies", e));
    let config = ferric_scf::AimdConfig {
        md,
        scf_warm_start: md_cfg.scf_warm_start.unwrap_or(true),
        masses_amu: None,
    };
    print_header(&config.md);
    let mut correction = |m: &Molecule, scf: &ferric_scf::result::ScfResult| match &model {
        Some(d) => {
            let (c, g) = d.evaluate(ctx, m, bs, op, rhf_config, scf)?;
            Ok((c.energy, Some(g)))
        }
        None => Ok((0.0, None)),
    };
    let correction: Option<&mut ferric_scf::ScfCorrection<'_>> = match model {
        Some(_) => Some(&mut correction),
        None => None,
    };
    let r = ferric_scf::run_aimd(
        ctx,
        mol,
        &bs.name,
        op,
        rhf_config,
        spin,
        &config,
        correction,
        &mut [&mut xyz, &mut csv, &mut FramePrinter],
    )
    .unwrap_or_else(|e| die(" during molecular dynamics", e));
    print_summary(&r, &xyz_path, &csv_path);
    let reference = crate::task_reference_certificate_from(
        method,
        cfg,
        ctx,
        bs,
        op,
        &r.mol,
        &r.last_scf,
        ferric_scf::gradient::gradient_task_config(rhf_config).into_owned(),
    );
    crate::print_and_note_certificate(&md_certificate(method, &r.md).with_reference(&reference));
    note_md_record(&r, &xyz_path, &csv_path);
}

fn print_header(md: &ferric_md::MdConfig) {
    println!(
        "\nMolecular dynamics: {} steps, dt = {} fs, thermostat = {}, seed = {}",
        md.n_steps,
        md.dt_fs,
        md.thermostat.as_str(),
        md.seed
    );
    println!(
        "  {:>6} {:>10} {:>18} {:>14} {:>18} {:>10}",
        "step", "time/fs", "E_pot/Eh", "E_kin/Eh", "conserved/Eh", "T/K"
    );
}

fn print_summary(r: &ferric_scf::AimdResult, xyz: &Path, csv: &Path) {
    let md = &r.md;
    println!("\nMolecular Dynamics Result:");
    println!("  surface         = {}", md.label);
    println!("  frames          = {}", md.frames.len());
    println!("  n_dof           = {}", md.n_dof);
    println!(
        "  final E_pot     = {:.10} Hartree",
        md.final_state.frame.potential
    );
    println!(
        "  max |drift|     = {:.3e} Hartree",
        md.max_conserved_drift()
    );
    println!("  mean T          = {:.2} K", md.mean_temperature(0));
    println!(
        "  SCF iterations  = {}",
        r.scf_iterations.iter().sum::<usize>()
    );
    if let Some(line) = &r.warm_start_summary {
        println!("{line}");
    }
    println!("  trajectory      = {}", xyz.display());
    println!("  energies        = {}", csv.display());
}

/// The ungraded MD certificate: capability `md:<kind>`, with the integration
/// settings as approximations and the drift as a warning.
fn md_certificate(method: &str, md: &MdResult) -> Certificate {
    let c = &md.config;
    let target = c.temperature_k.unwrap_or(f64::NAN);
    let (integrator, thermostat) = match c.thermostat {
        Thermostat::Nve => ("velocity Verlet", "none (NVE)".to_string()),
        Thermostat::Berendsen { tau_fs } => (
            "velocity Verlet",
            format!(
                "Berendsen, tau = {tau_fs} fs, T = {target} K (equilibration only; not canonical)"
            ),
        ),
        Thermostat::Langevin { friction_per_ps } => (
            "Langevin BAOAB",
            format!("Langevin, friction = {friction_per_ps} ps^-1, T = {target} K"),
        ),
    };
    Certificate::ungraded(format!("md:{method}"))
        .with_approximation("integrator", integrator)
        .with_approximation("thermostat", thermostat)
        .with_approximation(
            "time-step",
            format!("dt = {} fs, {} steps", c.dt_fs, c.n_steps),
        )
        .with_approximation("seed", c.seed.to_string())
        .warn(format!(
            "molecular dynamics has no validation grade in ferric; max conserved-quantity \
             drift {:.3e} Hartree over the run",
            md.max_conserved_drift()
        ))
}

/// The `md` run-log record: the files written and the run's summary numbers.
fn note_md_record(r: &ferric_scf::AimdResult, xyz: &Path, csv: &Path) {
    let Some(rl) = ferric_scf::runlog::log() else {
        return;
    };
    let md = &r.md;
    rl.note(
        "md",
        serde_json::json!({
            "surface": md.label,
            "n_steps": md.config.n_steps,
            "dt_fs": md.config.dt_fs,
            "thermostat": md.config.thermostat.as_str(),
            "seed": md.config.seed,
            "n_frames": md.frames.len(),
            "n_dof": md.n_dof,
            "final_potential": ferric_scf::runlog::json_f64(md.final_state.frame.potential),
            "max_conserved_drift": ferric_scf::runlog::json_f64(md.max_conserved_drift()),
            "scf_iterations": r.scf_iterations,
            "trajectory": xyz.display().to_string(),
            "energies": csv.display().to_string(),
        }),
    );
}
