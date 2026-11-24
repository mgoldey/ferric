//! `save_reference=` / `read_reference=` / `read_reference_other_build=` on
//! `run_rimp2`, `run_ccsd` and `run_ccsd_t`: the CLI's `[scf]` keys of the
//! same names (`ferric_cli::reference_io`), on the reference SCF these
//! entry points solve (`PreparedSystem::solve_rhf`, or `solve_uhf` for an
//! open-shell molecule).
//!
//! The file is the CLI's checkpoint, with the route field naming the Python
//! solve ("python solve_rhf" / "python solve_uhf"), so a Python file and a
//! CLI file are never read as each other. A read reference replaces the
//! solve; every refusal (another calculation, a corrupted or missing file,
//! another build) is an `InputError` (a `ValueError`) naming why, never a
//! fresh SCF in its place.

use crate::errors::{from_core, input_err};
use ferric_cli::reference_io::{self, ReferencePlan, Surface};
use ferric_core::certificate::Certificate;
use ferric_core::FerricError;
use ferric_scf::result::{ScfResult, Spin};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::PreparedSystem;
use pyo3::PyErr;
use pyo3::PyResult;

/// The three keywords, validated before any integral is computed.
#[derive(Clone, Debug)]
pub(crate) struct ReferenceFile {
    save: Option<String>,
    read: Option<String>,
    other_build: bool,
}

/// A driver failure: a core error, or a reference file this call cannot use.
pub(crate) enum DriverError {
    Core(FerricError),
    Reference(String),
}

impl From<FerricError> for DriverError {
    fn from(e: FerricError) -> Self {
        DriverError::Core(e)
    }
}

impl From<DriverError> for PyErr {
    fn from(e: DriverError) -> Self {
        match e {
            DriverError::Core(e) => from_core(&e),
            DriverError::Reference(msg) => input_err(msg),
        }
    }
}

impl ReferenceFile {
    /// `None` when neither path is given. `who` prefixes the refusals.
    pub(crate) fn new(
        who: &str,
        save: Option<String>,
        read: Option<String>,
        other_build: bool,
    ) -> PyResult<Option<Self>> {
        let refuse = |why: &str| Err(input_err(format!("{who}: {why}")));
        if other_build && read.is_none() {
            return refuse("read_reference_other_build is read only with read_reference");
        }
        if save.is_some() && read.is_some() {
            return refuse(
                "save_reference and read_reference cannot both be given: a read reference \
                 is already in its file",
            );
        }
        for (key, path) in [("save_reference", &save), ("read_reference", &read)] {
            if path.as_deref().is_some_and(|p| p.trim().is_empty()) {
                return refuse(&format!("{key} is an empty string"));
            }
        }
        Ok((save.is_some() || read.is_some()).then_some(ReferenceFile {
            save,
            read,
            other_build,
        }))
    }

    /// The reference of `sys` under `config`: read from the file, or solved by
    /// `solve` (and saved when asked). The second value is the certificate
    /// route of a read reference (`with_read_route`).
    pub(crate) fn solve(
        &self,
        sys: &PreparedSystem,
        config: &RhfConfig,
        solve: impl FnOnce() -> Result<ScfResult, FerricError>,
    ) -> Result<(ScfResult, Option<String>), DriverError> {
        let msg = |e: ferric_cli::error::CliError| DriverError::Reference(e.message());
        let (spin, route) = if sys.mol().multiplicity > 1 {
            (Spin::Unrestricted, "python solve_uhf")
        } else {
            (Spin::Restricted, "python solve_rhf")
        };
        let plan = ReferencePlan {
            spin,
            config,
            route: route.to_string(),
            surface: Surface::Python,
        };
        let identity = reference_io::identity(sys, &plan).map_err(msg)?;
        if let Some(path) = &self.save {
            reference_io::check_save_path(Surface::Python, path).map_err(msg)?;
        }
        let key = if self.read.is_some() {
            "read_reference"
        } else {
            "save_reference"
        };
        reference_io::refuse_unstorable(Surface::Python, key, config, config.k_builder)
            .map_err(msg)?;
        if let Some(path) = &self.read {
            let read =
                reference_io::read(path, sys, &identity, &plan, self.other_build).map_err(msg)?;
            return Ok((read.result, Some(read.provenance)));
        }
        let result = solve()?;
        if let Some(path) = &self.save {
            reference_io::save(path, sys, identity, &plan, &result).map_err(msg)?;
        }
        Ok((result, None))
    }
}

/// [`ReferenceFile::solve`] when a file is named, else `solve` alone.
pub(crate) fn solve_reference(
    file: Option<&ReferenceFile>,
    sys: &PreparedSystem,
    config: &RhfConfig,
    solve: impl FnOnce() -> Result<ScfResult, FerricError>,
) -> Result<(ScfResult, Option<String>), DriverError> {
    match file {
        Some(f) => f.solve(sys, config, solve),
        None => Ok((solve()?, None)),
    }
}

/// `c` with the CLI's `reference` route (`read from <file> (checksum ...)`)
/// when the reference was read.
pub(crate) fn with_read_route(c: Certificate, provenance: Option<String>) -> Certificate {
    match provenance {
        Some(p) => c.with_route("reference", p),
        None => c,
    }
}
