//! `[scf] save_reference` / `read_reference`: the reference SCF of a
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

use crate::error::CliError;
use ferric_scf::checkpoint::{
    BuildStamp, Checkpoint, FieldClass, IdentityField, ReadMode, ScfIdentity,
};
use ferric_scf::result::{ScfResult, Spin};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::PreparedSystem;
use std::path::Path;

/// How this run would solve its reference: the spin treatment, the
/// configuration the solver receives, and the route that converges it.
pub(crate) struct ReferencePlan<'a> {
    /// The reference's spin treatment.
    pub spin: Spin,
    /// The configuration handed to the solver (the open-shell routes change
    /// `mom_after_iter` on a copy of the run's configuration).
    pub config: &'a RhfConfig,
    /// Which driver converges it, with what the driver reads besides
    /// `config` (the DF-guess aux, for example). Compared as text.
    pub route: String,
}

/// Identity field names the CLI adds to the library's.
const ROUTE_FIELD: &str = "cli_reference_route";
const RANKS_FIELD: &str = "mpi_ranks";

/// The identity of the reference this run would solve.
pub(crate) fn identity(
    sys: &PreparedSystem,
    plan: &ReferencePlan<'_>,
) -> Result<ScfIdentity, CliError> {
    let mut id = ScfIdentity::new(
        sys.mol(),
        sys.basis_set(),
        plan.spin,
        &sys.op(),
        plan.config,
    )
    .map_err(|e| CliError::msg(format!("[scf] reference identity: {e}")))?;
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
/// save and recomputed on read, an independent check that the file's density
/// belongs to this basis and geometry.
fn energy_pieces(sys: &PreparedSystem, result: &ScfResult) -> (f64, f64) {
    let h = ferric_integrals::oneelectron::hcore(sys.basis());
    let e1 = (&result.density_total * &h).sum();
    (e1, sys.mol().nuclear_repulsion())
}

/// Before the SCF: the directory a `save_reference` file goes in must exist,
/// so a typo costs nothing rather than the whole SCF.
pub(crate) fn check_save_path(path: &str) -> Result<(), CliError> {
    let parent = Path::new(path)
        .parent()
        .filter(|p| !p.as_os_str().is_empty());
    if let Some(dir) = parent {
        if !dir.is_dir() {
            return Err(CliError::msg(format!(
                "[scf] save_reference {path:?}: directory {} does not exist",
                dir.display()
            )));
        }
    }
    Ok(())
}

/// Write the converged reference. A reference that did not converge is not
/// written (no read could use it), with a warning; the run goes on.
pub(crate) fn save(
    path: &str,
    sys: &PreparedSystem,
    identity: ScfIdentity,
    result: &ScfResult,
) -> Result<(), CliError> {
    if !result.converged {
        eprintln!(
            "warning: [scf] save_reference {path:?} not written: the reference SCF did not \
             converge (exit {:?}), and a read accepts only a converged reference",
            result.exit
        );
        return Ok(());
    }
    let (e1, vnn) = energy_pieces(sys, result);
    let checksum = Checkpoint::from_result(result, e1, vnn, identity, build_stamp())
        .and_then(|ck| ck.write(Path::new(path)))
        .map_err(|e| CliError::msg(format!("[scf] save_reference {path:?}: {e}")))?;
    eprintln!("[ferric] SCF reference saved to {path} (checksum {checksum:016x})");
    if let Some(rl) = ferric_scf::runlog::log() {
        rl.note(
            "scf_reference_saved",
            serde_json::json!({ "path": path, "checksum": format!("{checksum:016x}") }),
        );
    }
    Ok(())
}

/// A reference read from a file, and what the run log records about it.
pub(crate) struct ReadReference {
    pub result: ScfResult,
    /// The run log's `scf_reference` entry of the `run_end` record.
    pub record: serde_json::Value,
}

/// Read `path` in place of the SCF this run would solve under `plan`.
pub(crate) fn read(
    path: &str,
    sys: &PreparedSystem,
    current: &ScfIdentity,
    plan: &ReferencePlan<'_>,
    allow_other_build: bool,
) -> Result<ReadReference, CliError> {
    let fail = |e: String| CliError::msg(format!("[scf] read_reference {path:?}: {e}"));
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
                }
                ferric_scf::checkpoint::CheckpointError::OtherBuild { .. } => {
                    "; set [scf] read_reference_other_build = true to accept it"
                }
                _ => "",
            };
            fail(format!("{e}{hint}"))
        })?;
    let result = ck.restore(plan.config).map_err(|e| fail(e.to_string()))?;
    let state = ck
        .final_state
        .as_ref()
        .ok_or_else(|| fail("not a final file".to_string()))?;
    let (e1, vnn) = energy_pieces(sys, &result);
    if e1.to_bits() != state.one_electron_energy.to_bits()
        || vnn.to_bits() != state.nuclear_repulsion.to_bits()
    {
        return Err(fail(format!(
            "the stored density does not reproduce its one-electron energy on this run's \
             integrals (Tr[Dh] {e1:e} vs stored {:e}, Vnn {vnn:e} vs stored {:e})",
            state.one_electron_energy, state.nuclear_repulsion
        )));
    }
    let checksum = verdict.file_hash.map(|h| format!("{h:016x}"));
    eprintln!(
        "[ferric] SCF reference read from {path} (checksum {}, {} iterations when it was \
         solved, exit {:?}); no SCF iterations run",
        checksum.as_deref().unwrap_or("none"),
        result.iterations,
        result.exit,
    );
    if verdict.build_override_used {
        eprintln!(
            "warning: [scf] read_reference: the file was written by build {}, this is {}; \
             accepted on read_reference_other_build",
            verdict.file_build,
            build_stamp()
        );
    }
    let record = serde_json::json!({
        "read": path,
        "checksum": checksum,
        "file_iterations": result.iterations,
        "file_build": verdict.file_build.to_string(),
        "build_override_used": verdict.build_override_used,
    });
    Ok(ReadReference { result, record })
}
