//! `AoIntegrals.provenance()` and `certificate_from_blocks`: the record of
//! what an `AoIntegrals` object computed, and the certificate a caller's
//! script composes from such records
//! (`ferric_core::certificate::blocks`). Recording never touches an array.

use ferric_core::basis::BasisSet;
use ferric_core::certificate::{BasisRecord, BlockRecord, BlockUse, BuildStamp, Certificate};
use ferric_core::mol::Molecule;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The build this extension was compiled from.
fn this_build() -> BuildStamp {
    BuildStamp::new(
        ferric_build_info::VERSION,
        ferric_build_info::COMMIT,
        ferric_build_info::DIRTY,
    )
}

/// Distinct atomic numbers of `mol`, ghosts included (they carry basis
/// functions).
fn elements(mol: &Molecule) -> Vec<i32> {
    mol.atoms.iter().map(|a| a.z).collect()
}

/// The record of one `AoIntegrals` object, behind a mutex so `&self`
/// methods can count their calls.
pub(super) struct Log {
    record: std::sync::Mutex<BlockRecord>,
    elements: Vec<i32>,
}

impl Log {
    /// The empty record of an object on `mol` (ECP charges applied) in `bs`.
    pub(super) fn new(mol: &Molecule, bs: &BasisSet) -> Self {
        let elements = elements(mol);
        let mut record =
            BlockRecord::new("AoIntegrals", this_build(), BasisRecord::of(bs, &elements));
        let mut ecp: Vec<i32> = mol
            .atoms
            .iter()
            .filter(|a| a.n_core_ecp > 0)
            .map(|a| a.z)
            .collect();
        ecp.sort_unstable();
        ecp.dedup();
        if !ecp.is_empty() {
            record = record.with_approximation("ecp", format!("{} ECP on Z = {ecp:?}", bs.name));
        }
        Log {
            record: std::sync::Mutex::new(record),
            elements,
        }
    }

    /// Count one call of `u`.
    pub(super) fn note(&self, u: BlockUse) {
        self.lock().record_use(u);
    }

    /// The record of the auxiliary basis `aux` on this object's elements.
    pub(super) fn aux(&self, aux: &BasisSet) -> BasisRecord {
        BasisRecord::of(aux, &self.elements)
    }

    /// A copy of the record so far.
    pub(super) fn snapshot(&self) -> BlockRecord {
        self.lock().clone()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, BlockRecord> {
        self.record.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The operator as a record names it: `"coulomb"`, or `"erf omega=0.3"`.
/// Called after `parse_operator` accepted the pair.
pub(super) fn op_text(operator: &str, omega: Option<f64>) -> String {
    match omega {
        Some(w) => format!("{operator} omega={w}"),
        None => operator.to_string(),
    }
}

/// `"<functional> on <main grid>"`, plus the NLC grid for a VV10 functional,
/// in the SCF certificate's grid text.
pub(super) fn xc_text(name: &str, grids: &ferric_scf::ks_gradient::KsGradGrids) -> String {
    let main = ferric_scf::certificate::grid_text(&grids.main);
    let vv10 = ferric_dft::libxc::xc_def_from_name(name).is_ok_and(|d| d.vv10.is_some());
    if vv10 {
        let nlc = ferric_scf::certificate::grid_text(&grids.nlc);
        format!("{name} on {main}, NLC grid {nlc}")
    } else {
        format!("{name} on {main}")
    }
}

/// JSON text as a Python object (dicts, lists, str, numbers, None).
pub(super) fn json_to_py(py: Python<'_>, json: &str) -> PyResult<PyObject> {
    Ok(py.import("json")?.call_method1("loads", (json,))?.unbind())
}

/// A Python object (a dict from `provenance()` or `.certificate`) as JSON
/// text.
fn py_to_json(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<String> {
    py.import("json")?.call_method1("dumps", (obj,))?.extract()
}

/// The certificate of a result composed from building blocks.
///
/// `records` are `AoIntegrals.provenance()` dicts. The certificate's
/// capability is `"blocks:<capability>"` and its grade `"not graded"`,
/// whatever the records and `reference` hold. `blocks` carries the records;
/// the approximations are the ones the records imply (`ri` per auxiliary
/// basis, `operator`, `xc-grid`, `screening`, `ecp`, ...), then those of
/// `reference` (a driver certificate dict, prefixed `scf `), then
/// `approximations`. Records from different builds add a warning.
/// `ValueError` on an empty or malformed record list.
#[pyfunction]
#[pyo3(signature = (capability, records, approximations=Vec::new(), warnings=Vec::new(), reference=None))]
pub(super) fn certificate_from_blocks(
    py: Python<'_>,
    capability: &str,
    records: Vec<Bound<'_, PyAny>>,
    approximations: Vec<(String, String)>,
    warnings: Vec<String>,
    reference: Option<Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let records = records
        .iter()
        .enumerate()
        .map(|(i, r)| {
            BlockRecord::from_json_str(&py_to_json(py, r)?)
                .map_err(|e| PyValueError::new_err(format!("records[{i}]: {e}")))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let mut c = Certificate::from_blocks(capability, records).map_err(PyValueError::new_err)?;
    if let Some(r) = reference {
        let r = Certificate::from_json_str(&py_to_json(py, &r)?)
            .map_err(|e| PyValueError::new_err(format!("reference: {e}")))?;
        c = c.with_reference(&r);
    }
    for (name, detail) in approximations {
        c = c.with_approximation(name, detail);
    }
    c.extend_warnings(warnings);
    json_to_py(py, &c.to_json().to_string())
}
