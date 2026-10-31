//! SCF orbital checkpoint: file format, atomic IO, identity checks.
//!
//! NO SOLVER READS OR WRITES THIS YET. It is the first of the five steps in the
//! checkpoint design, and it exists so that the later steps have one reviewed
//! answer to "is this file the calculation I am about to trust?".
//!
//! # File
//!
//! An uncompressed NPZ, readable by `numpy.load(path, allow_pickle=False)`:
//!
//! | member | dtype | contents |
//! |---|---|---|
//! | `format_version` | `uint64`, shape (1,) | [`FORMAT_VERSION`] |
//! | `checksum` | `uint64`, shape (1,) | FNV-1a 64, defined below |
//! | `meta` | `uint8` | UTF-8 JSON: build, identity, spin, exit and scalars |
//! | `density_total`, `density_alpha`, `mos_alpha`, `eps_alpha` | `float64` | always |
//! | `density_beta`, `mos_beta`, `eps_beta` | `float64` | open shell |
//! | `fock_alpha`, `fock_beta`, `rohf_spin_fock_alpha`, `rohf_spin_fock_beta` | `float64` | final file only |
//!
//! The checksum runs over the bytes of `ferric-scf-checkpoint\0`, the version
//! (u64 LE), the `meta` length (u64 LE) and bytes, and then each float array in
//! byte-wise name order: name length (u64 LE), name, ndim (u64 LE), each
//! dimension (u64 LE), one byte that is 1 for Fortran order, and the values as
//! f64 LE in memory order. `crates/ferric-python/tests/test_scf_checkpoint_format.py`
//! recomputes it from this paragraph, with numpy as the reader.
//!
//! The layout flag is part of the checksum because layout is part of what
//! round-trips: [`Checkpoint::read`] returns an array in the memory order it was
//! written in, so code that flat-indexes a matrix behaves the same on a restored
//! one. (`ndarray` GEMM output and LAPACK eigenvectors are not always
//! row-major.) An array that is neither C- nor F-contiguous is rewritten to C
//! order on the way out; its values are unchanged.
//!
//! # What identifies a calculation
//!
//! [`ScfIdentity`] holds named canonical fields, not one hash, so that an error
//! can say what differs. `BasisSet.shells` and `BasisSet.ecps` are `HashMap`s
//! whose `{:?}` order changes from one process to the next, so a key must never
//! be built from the `Debug` form of a `BasisSet`: the basis enters per element
//! of the molecule, ascending, as a hash of its explicit contents.
//!
//! [`ScfIdentity::new`] destructures `RhfConfig` with no `..`, and
//! [`Checkpoint::from_result`] does the same for `ScfResult`, so a new field on
//! either does not compile until someone decides what it means here.
//!
//! Identity also covers process-wide state outside `RhfConfig`: the libint
//! precision, ECP backend, linear-dependence threshold, GPU mode and precision,
//! libint version, cargo features, and the environment switches the SCF reads.
//!
//! # What a mismatch does ([`Checkpoint::check`])
//!
//! | | guess | skip |
//! |---|---|---|
//! | [`FieldClass::Structure`] | error | error |
//! | [`FieldClass::Coordinates`] | warn, with the max shift | error, bitwise |
//! | [`FieldClass::Environment`], [`FieldClass::Method`] | warn | error |
//! | [`FieldClass::StateSelecting`] | warn | error |
//! | [`FieldClass::Route`] | log | error |
//! | `energy_conv`, `density_conv` | ignored | file at least as tight |
//! | file exit state | any | `Converged` or `Plateau` (echoed), with `converged = true` |
//! | build | log | error unless overridden, and the override is recorded |
//!
//! A read that fails is always `Err`: there is no return value that means "start
//! fresh instead", and a missing file is [`CheckpointError::Missing`].

mod format;
mod identity;
#[cfg(test)]
mod tests;

pub use identity::{
    BuildStamp, FieldClass, IdentityField, Mismatch, ReadMode, ReadVerdict, ScfIdentity,
};

use crate::result::{ConvergenceReport, DfJkRoute, ScfExit, ScfResult, Spin};
use ferric_core::FerricError;
use ndarray::Array2;
use std::path::Path;

/// Version of the on-disk layout. Bump it for any change a v1 reader would
/// misread; the frozen file in `testdata/checkpoint/` fails if the layout moves.
pub const FORMAT_VERSION: u64 = 1;

/// Why a checkpoint could not be read, written or trusted.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CheckpointError {
    #[error("checkpoint {path:?} does not exist")]
    Missing { path: String },
    #[error("checkpoint {path:?}: {detail}")]
    Io { path: String, detail: String },
    #[error("checkpoint is not readable: {0}")]
    Format(String),
    #[error(
        "checkpoint format_version is {found}; this build reads {}",
        FORMAT_VERSION
    )]
    Version { found: u64 },
    #[error("checkpoint checksum mismatch: stored {stored:016x}, computed {computed:016x}")]
    Checksum { stored: u64, computed: u64 },
    #[error("checkpoint {array}: expected shape {expected}, found {found}")]
    Shape {
        array: String,
        expected: String,
        found: String,
    },
    #[error("checkpoint {array} holds a non-finite value")]
    NonFinite { array: String },
    #[error("checkpoint is for a different calculation: {}", identity::describe(.0))]
    Identity(Vec<Mismatch>),
    #[error("checkpoint was written by build {file}, this is {current}; skipping the SCF across builds needs the explicit override")]
    OtherBuild { file: String, current: String },
    #[error("checkpoint exit state {exit} cannot replace an SCF (Converged or Plateau only)")]
    ExitRefused { exit: String },
    #[error("checkpoint is a mid-run file, which can seed an SCF but not replace one")]
    MidRun,
    #[error("checkpoint {field} = {file:e} is looser than the requested {requested:e}")]
    Looser {
        field: String,
        file: f64,
        requested: f64,
    },
    #[error(
        "checkpoint state carries {0:?}, which it does not store, so it cannot replace an SCF"
    )]
    NotRestorable(Vec<String>),
}

impl From<CheckpointError> for FerricError {
    fn from(e: CheckpointError) -> Self {
        FerricError::General(e.to_string())
    }
}

/// The orbitals and densities, present in every file.
#[derive(Debug, Clone)]
pub struct OrbitalState {
    pub spin: Spin,
    pub density_total: Array2<f64>,
    pub density_alpha: Array2<f64>,
    pub density_beta: Option<Array2<f64>>,
    pub mos_alpha: Array2<f64>,
    pub mos_beta: Option<Array2<f64>>,
    pub eps_alpha: Vec<f64>,
    pub eps_beta: Option<Vec<f64>>,
}

/// What only a finished SCF has. A mid-run file has none of it: its Fock matrix
/// may carry a level shift, so it is not stored.
#[derive(Debug, Clone)]
pub struct FinalState {
    pub energy: f64,
    /// `Tr[D h]`, for the independent check a skip runs against freshly computed
    /// one-electron integrals. The caller supplies it: `ScfResult` does not hold
    /// it.
    pub one_electron_energy: f64,
    pub nuclear_repulsion: f64,
    pub fock_alpha: Array2<f64>,
    pub fock_beta: Option<Array2<f64>>,
    pub rohf_spin_focks: Option<(Array2<f64>, Array2<f64>)>,
    /// `None` means exact four-centre J/K. Dropping a route here would pair an
    /// RI energy with an exact gradient, silently.
    pub df_jk: Option<DfJkRoute>,
    pub exit: ScfExit,
    pub converged: bool,
    pub iterations: usize,
    pub computed_quartets: usize,
    pub convergence: Option<ConvergenceReport>,
    /// `PrecisionRecord::summary()`, for echoing. The record holds `&'static str`
    /// fields, so it is kept as text: a restored mixed-precision run can say what
    /// ran but cannot rebuild the typed record.
    pub precision_summary: String,
    /// `{:?}` of the same record.
    pub precision_detail: String,
    /// Populated `ScfResult` fields this file does not store. `induced_dipoles`
    /// and `cosx_final` here make the state unusable under skip.
    pub not_stored: Vec<String>,
}

/// One checkpoint: the file's contents, independent of its bytes.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub build: BuildStamp,
    pub identity: ScfIdentity,
    pub orbitals: OrbitalState,
    /// `None` for a mid-run file.
    pub final_state: Option<FinalState>,
    /// SCF iterations behind the state; for a final file, also
    /// [`FinalState::iterations`].
    pub iteration: usize,
    /// The checksum read from disk; `None` until a file has been read.
    pub file_hash: Option<u64>,
}

impl Checkpoint {
    /// A finished SCF. `one_electron_energy` is `Tr[D h]` and `nuclear_repulsion`
    /// is `Vnn`, both from the solve that produced `result`.
    pub fn from_result(
        result: &ScfResult,
        one_electron_energy: f64,
        nuclear_repulsion: f64,
        identity: ScfIdentity,
        build: BuildStamp,
    ) -> Result<Self, CheckpointError> {
        // No `..`: a new ScfResult field must be classified here to compile.
        let ScfResult {
            spin,
            energy,
            density_total,
            density_alpha,
            density_beta,
            mos_alpha,
            mos_beta,
            eps_alpha,
            eps_beta,
            fock_alpha,
            fock_beta,
            converged,
            exit,
            iterations,
            computed_quartets,
            df_jk,
            rohf_spin_focks,
            convergence,
            precision,
            // Not stored; a skip must refuse a state that had these.
            induced_dipoles,
            cosx_final,
            // Not stored; a restored result reports them as not run.
            stability,
            stability_external,
            cosx_schedule,
            guess_basis,
        } = result;
        let not_stored = [
            ("induced_dipoles", induced_dipoles.is_some()),
            ("cosx_final", cosx_final.is_some()),
            ("stability", stability.is_some()),
            ("stability_external", stability_external.is_some()),
            ("cosx_schedule", cosx_schedule.is_some()),
            ("guess_basis", guess_basis.is_some()),
        ]
        .iter()
        .filter(|(_, present)| *present)
        .map(|(name, _)| (*name).to_string())
        .collect();
        let ck = Self {
            build,
            identity,
            orbitals: OrbitalState {
                spin: *spin,
                density_total: density_total.clone(),
                density_alpha: density_alpha.clone(),
                density_beta: density_beta.clone(),
                mos_alpha: mos_alpha.clone(),
                mos_beta: mos_beta.clone(),
                eps_alpha: eps_alpha.clone(),
                eps_beta: eps_beta.clone(),
            },
            final_state: Some(FinalState {
                energy: *energy,
                one_electron_energy,
                nuclear_repulsion,
                fock_alpha: fock_alpha.clone(),
                fock_beta: fock_beta.clone(),
                rohf_spin_focks: rohf_spin_focks.clone(),
                df_jk: df_jk.clone(),
                exit: *exit,
                converged: *converged,
                iterations: *iterations,
                computed_quartets: *computed_quartets,
                convergence: *convergence,
                precision_summary: precision.summary(),
                precision_detail: format!("{precision:?}"),
                not_stored,
            }),
            iteration: *iterations,
            file_hash: None,
        };
        ck.validate()?;
        Ok(ck)
    }

    /// A mid-run state, usable as a guess and never as a replacement for an SCF.
    pub fn mid_run(
        orbitals: OrbitalState,
        iteration: usize,
        identity: ScfIdentity,
        build: BuildStamp,
    ) -> Result<Self, CheckpointError> {
        let ck = Self {
            build,
            identity,
            orbitals,
            final_state: None,
            iteration,
            file_hash: None,
        };
        ck.validate()?;
        Ok(ck)
    }

    /// Write atomically: same-directory temp file, `sync_all`, rename, fsync of
    /// the directory. At every instant `path` holds the old complete file or the
    /// new complete file. A kill can leave a `.<name>.tmp.<pid>.<seq>` sibling
    /// behind; nothing reads those, and the next write to `path` removes any no
    /// live writer holds locked. A symlink at `path` is refused, because the
    /// rename would replace the link and leave its target stale. Returns the
    /// checksum.
    pub fn write(&self, path: &Path) -> Result<u64, CheckpointError> {
        format::write(self, path, &mut |_| {})
    }

    /// Read and verify a file: version, checksum, member set, shapes, finiteness.
    /// Every failure is an `Err`, so a caller cannot fall back by accident.
    pub fn read(path: &Path) -> Result<Self, CheckpointError> {
        format::read(path)
    }

    /// Shapes, spin consistency and finiteness. Run on both write and read.
    pub(crate) fn validate(&self) -> Result<(), CheckpointError> {
        format::validate(self)
    }

    /// Decide whether this file may be used for the calculation described by
    /// `current`. `Err` is a hard stop; warnings and log lines come back in the
    /// [`ReadVerdict`] for the caller to print and record.
    pub fn check(
        &self,
        current: &ScfIdentity,
        current_build: &BuildStamp,
        mode: ReadMode,
    ) -> Result<ReadVerdict, CheckpointError> {
        identity::check(self, current, current_build, mode)
    }
}
