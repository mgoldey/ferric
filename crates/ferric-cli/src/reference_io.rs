//! `[scf] save_reference` / `read_reference` (and the same keywords on the
//! Python correlated drivers, `Surface::Python`): the reference SCF of a
//! correlated kind, written once it converges and read back in place of the
//! solve by a later run (RI-MP2 now, CCSD on the same reference next).
//!
//! The file is a `ferric_scf::checkpoint` final file. A read is a
//! [`ReadMode::Skip`]: every identity field must match, coordinates bitwise,
//! the file must be `Converged` and at least as tight as this run asks, and it
//! must come from this build unless `read_reference_other_build` is set. A
//! mismatch is an error, never a fresh SCF in its place: a run that names a
//! file and quietly solves anyway would hide a wrong path.
//!
//! The identity is the library's ([`ScfIdentity::new`] on the configuration
//! the reference is solved with) plus what decides the reference in the CLI
//! and is not in `RhfConfig`: the route that converges it
//! ([`ReferencePlan::route`]: plain `solve_rhf`, the DF-guess or
//! DF-increments two-stage SCF and its aux, the UHF driver, GW's UHF/ROHF)
//! and the MPI rank count.
//!
//! The Python bindings call the same functions (`save_reference=` /
//! `read_reference=` on `run_rimp2`, `run_ccsd`, `run_ccsd_t`) with their own
//! route names, so a Python file is never read as a CLI one or the reverse;
//! [`Surface`] picks how a message names the keys.

use crate::error::CliError;
use ferric_scf::checkpoint::{
    BuildStamp, Checkpoint, FieldClass, IdentityField, ReadMode, ScfIdentity,
};
use ferric_scf::result::{ScfResult, Spin};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::PreparedSystem;
use std::path::Path;

/// Who names the keys: the CLI's `[scf]` table or a Python keyword.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Surface {
    /// `[scf] save_reference = "..."`.
    Cli,
    /// `save_reference="..."` on a Python entry point.
    Python,
}

impl Surface {
    /// The key as the user wrote it, for a message.
    fn key(self, name: &str) -> String {
        match self {
            Surface::Cli => format!("[scf] {name}"),
            Surface::Python => name.to_string(),
        }
    }

    /// The setting that turns `name` on, for a hint.
    fn set_true(self, name: &str) -> String {
        match self {
            Surface::Cli => format!("set [scf] {name} = true"),
            Surface::Python => format!("pass {name}=True"),
        }
    }
}

/// How this run would solve its reference: the spin treatment, the
/// configuration the solver receives, and the route that converges it.
pub struct ReferencePlan<'a> {
    /// The reference's spin treatment.
    pub spin: Spin,
    /// The configuration handed to the solver (the open-shell routes change
    /// `mom_after_iter` on a copy of the run's configuration).
    pub config: &'a RhfConfig,
    /// Which driver converges it, with what the driver reads besides
    /// `config` (the DF-guess aux, for example). Compared as text.
    pub route: String,
    /// How messages name the keys.
    pub surface: Surface,
}

/// Identity field names the CLI adds to the library's.
const ROUTE_FIELD: &str = "cli_reference_route";
const RANKS_FIELD: &str = "mpi_ranks";

/// The identity of the reference this run would solve.
pub fn identity(sys: &PreparedSystem, plan: &ReferencePlan<'_>) -> Result<ScfIdentity, CliError> {
    let mut id = ScfIdentity::new(
        sys.mol(),
        sys.basis_set(),
        plan.spin,
        &sys.op(),
        plan.config,
    )
    .map_err(|e| CliError::msg(format!("{}: {e}", plan.surface.key("reference identity"))))?;
    for (name, value) in [
        (ROUTE_FIELD, plan.route.clone()),
        (RANKS_FIELD, sys.ctx().size.to_string()),
    ] {
        id.fields.push(IdentityField {
            name: name.to_string(),
            class: FieldClass::Route,
            value,
        });
    }
    Ok(id)
}

/// The running build, as the checkpoint compares builds.
fn build_stamp() -> BuildStamp {
    BuildStamp {
        version: ferric_build_info::VERSION.to_string(),
        commit: ferric_build_info::commit_known().then(|| ferric_build_info::COMMIT.to_string()),
        dirty: ferric_build_info::DIRTY,
    }
}

/// `(Tr[D h], Vnn)` with `h = T + V` from this run's integrals: stored on
/// save and recomputed on read. A cheap consistency check that the file's
/// density fits this basis and geometry, on top of the identity (which
/// already compares both).
fn energy_pieces(sys: &PreparedSystem, density_total: &ndarray::Array2<f64>) -> (f64, f64) {
    let h = ferric_integrals::oneelectron::hcore(sys.basis());
    let e1 = (density_total * &h).sum();
    (e1, sys.mol().nuclear_repulsion())
}

/// Before the SCF: a `save_reference` path must name a file in a directory
/// that exists, not a directory or a symlink (the atomic write refuses a
/// symlink only after the SCF), so a typo costs nothing rather than the whole
/// SCF.
pub fn check_save_path(surface: Surface, path: &str) -> Result<(), CliError> {
    let key = surface.key("save_reference");
    let fail = |why: String| CliError::msg(format!("{key} {path:?}: {why}"));
    let p = Path::new(path);
    if let Ok(meta) = std::fs::symlink_metadata(p) {
        if meta.file_type().is_symlink() {
            return Err(fail("is a symlink; name the file itself".to_string()));
        }
        if meta.is_dir() {
            return Err(fail("is a directory; name a file in it".to_string()));
        }
    }
    let parent = p.parent().filter(|d| !d.as_os_str().is_empty());
    if let Some(dir) = parent {
        if !dir.is_dir() {
            return Err(fail(format!("directory {} does not exist", dir.display())));
        }
    }
    Ok(())
}

/// Before the SCF: refuse a reference the file cannot store, so a
/// `save_reference` never writes a file every read refuses and a
/// `read_reference` never pretends to replace such an SCF. Both are
/// properties of the solve itself: the COSX final-grid pass (its energy is
/// not the stored Fock's) and polarizable embedding (its induced dipoles are
/// not stored).
pub fn refuse_unstorable(
    surface: Surface,
    key: &str,
    config: &RhfConfig,
    k_builder: Option<ferric_scf::options::KBuilderKind>,
) -> Result<(), CliError> {
    let key = surface.key(key);
    if k_builder == Some(ferric_scf::options::KBuilderKind::Cosx)
        && config.cosx.final_grid.is_some()
    {
        let instead = match surface {
            Surface::Cli => "Set [scf] cosx_final_pass = false, or use another exchange route",
            Surface::Python => "Use another k_builder",
        };
        return Err(CliError::msg(format!(
            "{key} cannot be used with the COSX final-grid pass (on by default with \
             COSX exchange): the reference's energy comes from a one-shot pass on another \
             grid that the file does not store. {instead}"
        )));
    }
    if config.polarizable.is_some() {
        return Err(CliError::msg(format!(
            "{key} cannot be used with polarizable embedding ([qmmm] \
             polarizabilities_angstrom3): the converged induced dipoles are not stored"
        )));
    }
    Ok(())
}

/// Write the converged reference. A reference that did not converge is not
/// written (no read could use it), with a warning; the run goes on. Under MPI
/// only the root rank writes. `config` is the one the solver received: the
/// in-core ERI route it recorded is stored with the file.
pub fn save(
    path: &str,
    sys: &PreparedSystem,
    identity: ScfIdentity,
    plan: &ReferencePlan<'_>,
    result: &ScfResult,
) -> Result<(), CliError> {
    let (key, config) = (plan.surface.key("save_reference"), plan.config);
    if !sys.ctx().is_root() {
        return Ok(());
    }
    if !result.converged {
        eprintln!(
            "warning: {key} {path:?} not written: the reference SCF did not \
             converge (exit {:?}), and a read accepts only a converged reference",
            result.exit
        );
        return Ok(());
    }
    let (e1, vnn) = energy_pieces(sys, &result.density_total);
    let checksum = Checkpoint::from_result(result, e1, vnn, identity, build_stamp())
        .and_then(|mut ck| {
            if let Some(state) = ck.final_state.as_mut() {
                state.eri_route = config.eri_route.get();
            }
            ck.write(Path::new(path))
        })
        .map_err(|e| CliError::msg(format!("{key} {path:?}: {e}")))?;
    eprintln!("[ferric] SCF reference saved to {path} (checksum {checksum:016x})");
    if let Some(rl) = ferric_scf::runlog::log() {
        rl.note(
            "scf_reference_saved",
            serde_json::json!({ "path": path, "checksum": format!("{checksum:016x}") }),
        );
    }
    Ok(())
}

/// A reference read from a file, and what the run log and certificate record
/// about it.
pub struct ReadReference {
    pub result: ScfResult,
    /// The run log's `scf_reference` entry of the `run_end` record.
    pub record: serde_json::Value,
    /// The certificate route `reference`: which file, which checksum.
    pub provenance: String,
}

/// Read `path` in place of the SCF this run would solve under `plan`.
pub fn read(
    path: &str,
    sys: &PreparedSystem,
    current: &ScfIdentity,
    plan: &ReferencePlan<'_>,
    allow_other_build: bool,
) -> Result<ReadReference, CliError> {
    let surface = plan.surface;
    let key = surface.key("read_reference");
    let fail = |e: String| CliError::msg(format!("{key} {path:?}: {e}"));
    let ck = Checkpoint::read(Path::new(path)).map_err(|e| fail(e.to_string()))?;
    let verdict = ck
        .check(
            current,
            &build_stamp(),
            ReadMode::Skip { allow_other_build },
        )
        .map_err(|e| {
            let hint = match e {
                ferric_scf::checkpoint::CheckpointError::Identity(_) => {
                    "; this run would solve a different reference, so remove read_reference \
                     to solve it, or save one with these settings"
                        .to_string()
                }
                ferric_scf::checkpoint::CheckpointError::OtherBuild { .. } => {
                    format!(
                        "; {} to accept it",
                        surface.set_true("read_reference_other_build")
                    )
                }
                _ => String::new(),
            };
            fail(format!("{e}{hint}"))
        })?;
    let state = ck
        .final_state
        .as_ref()
        .ok_or_else(|| fail("not a final file".to_string()))?;
    let (stored_e1, stored_vnn) = (state.one_electron_energy, state.nuclear_repulsion);
    let writer_threads = state.writer_threads;
    let (file_energy_conv, file_density_conv) = (ck.identity.energy_conv, ck.identity.density_conv);
    // Consumed: the arrays move into the result.
    let result = ck
        .into_result(plan.config)
        .map_err(|e| fail(e.to_string()))?;
    let (e1, vnn) = energy_pieces(sys, &result.density_total);
    if e1.to_bits() != stored_e1.to_bits() || vnn.to_bits() != stored_vnn.to_bits() {
        return Err(fail(format!(
            "the stored density does not reproduce its one-electron energy on this run's \
             integrals (Tr[Dh] {e1:e} vs stored {stored_e1:e}, Vnn {vnn:e} vs stored \
             {stored_vnn:e})"
        )));
    }
    let checksum = verdict.file_hash.map(|h| format!("{h:016x}"));
    let checksum_text = checksum.as_deref().unwrap_or("none");
    eprintln!(
        "[ferric] SCF reference read from {path} (checksum {checksum_text}, {} iterations \
         when it was solved, exit {:?}); no SCF iterations run",
        result.iterations, result.exit,
    );
    if file_energy_conv < current.energy_conv || file_density_conv < current.density_conv {
        eprintln!(
            "[ferric] the reference was converged tighter than this run asks (energy_conv \
             {file_energy_conv:e}, density_conv {file_density_conv:e}); the results are those \
             of that reference, not of a fresh SCF at this run's thresholds"
        );
    }
    if verdict.build_override_used {
        eprintln!(
            "warning: {key}: the file was written by build {}, this is {}; \
             accepted on read_reference_other_build",
            verdict.file_build,
            build_stamp()
        );
    }
    let record = serde_json::json!({
        "read": path,
        "checksum": checksum,
        "file_iterations": result.iterations,
        "file_energy_conv": file_energy_conv,
        "file_density_conv": file_density_conv,
        "file_threads": writer_threads,
        "file_build": verdict.file_build.to_string(),
        "build_override_used": verdict.build_override_used,
    });
    let provenance = format!("read from {path} (checksum {checksum_text})");
    Ok(ReadReference {
        result,
        record,
        provenance,
    })
}
