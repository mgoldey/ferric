//! The `ferric` exception hierarchy and the one place that decides which
//! class a failure raises.
//!
//! Every class derives from [`FerricError`](`ferric.FerricError`) **and** from
//! the builtin class the same failure raised before the hierarchy existed, so
//! `except RuntimeError` / `except ValueError` / `except NotImplementedError`
//! handlers keep catching exactly what they caught. The classes are built with
//! `type(name, bases, ns)` at module initialisation (`create_exception!`
//! supports a single base), get `__module__ = "ferric"` and are attributes of
//! the `ferric` module, which is what makes them picklable.
//!
//! | class | bases | raised for |
//! |---|---|---|
//! | `FerricError` | `Exception` | root; never raised itself |
//! | `ConvergenceError` | `FerricError`, `RuntimeError` | `ScfConvergence`, `Convergence` |
//! | `BasisError` | `InputError`, `RuntimeError` | `Basis` |
//! | `NumericalError` | `FerricError`, `RuntimeError` | `Lapack` |
//! | `MemoryBudgetError` | `FerricError`, `RuntimeError` | `MemoryExhausted` |
//! | `UnsupportedError` | `FerricError`, `NotImplementedError` | `Unsupported` |
//! | `ComputationError` | `FerricError`, `RuntimeError` | every other `FerricError` and binding-level runtime failure |
//! | `InputError` | `FerricError`, `ValueError` | binding-level invalid arguments |
//!
//! `BasisError` is the one class with both builtins: the same bad basis was a
//! `ValueError` from `BasisSet.bundled` / `BasisSet.n_functions` and a
//! `RuntimeError` from the `run_*` drivers, and each handler must keep
//! catching what it caught.

use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::{PyDict, PyTuple, PyType};

use ferric_core::{ErrorCategory, FerricError as CoreError};

/// Which Python class a failure maps to.
#[derive(Clone, Copy)]
pub(crate) enum Kind {
    Convergence,
    Numerical,
    MemoryBudget,
    Unsupported,
    Computation,
    Input,
    // After `Input`: its class derives from `InputError`.
    Basis,
}

impl Kind {
    const ALL: [Kind; 7] = [
        Kind::Convergence,
        Kind::Numerical,
        Kind::MemoryBudget,
        Kind::Unsupported,
        Kind::Computation,
        Kind::Input,
        Kind::Basis,
    ];

    /// Position in [`Kind::ALL`] / the created-class table.
    fn idx(self) -> usize {
        match self {
            Kind::Convergence => 0,
            Kind::Numerical => 1,
            Kind::MemoryBudget => 2,
            Kind::Unsupported => 3,
            Kind::Computation => 4,
            Kind::Input => 5,
            Kind::Basis => 6,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Kind::Convergence => "ConvergenceError",
            Kind::Basis => "BasisError",
            Kind::Numerical => "NumericalError",
            Kind::MemoryBudget => "MemoryBudgetError",
            Kind::Unsupported => "UnsupportedError",
            Kind::Computation => "ComputationError",
            Kind::Input => "InputError",
        }
    }

    fn builtin(self) -> &'static str {
        match self {
            Kind::Unsupported => "NotImplementedError",
            Kind::Input => "ValueError",
            _ => "RuntimeError",
        }
    }

    fn doc(self) -> &'static str {
        match self {
            Kind::Convergence => {
                "An iterative procedure (SCF, a response or CC solver) did not converge."
            }
            Kind::Basis => "A basis set or ECP could not be loaded or does not cover the molecule (also a `ValueError` and a `RuntimeError`).",
            Kind::Numerical => "A dense linear-algebra call (eigensolver, factorization) failed.",
            Kind::MemoryBudget => "A memory budget or the shared pool cannot cover a request.",
            Kind::Unsupported => "A valid option combination the method does not implement.",
            Kind::Computation => "Any other failure inside a computation (also a `RuntimeError`).",
            Kind::Input => "An invalid argument or option value (also a `ValueError`).",
        }
    }
}

static ROOT: GILOnceCell<Py<PyType>> = GILOnceCell::new();
static CLASSES: GILOnceCell<Vec<Py<PyType>>> = GILOnceCell::new();

fn root(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    let t = ROOT.get_or_try_init(py, || -> PyResult<Py<PyType>> {
        let ns = PyDict::new(py);
        ns.set_item("__module__", "ferric")?;
        ns.set_item(
            "__doc__",
            "Root of the ferric exception hierarchy. Every subclass also derives from the builtin class the failure raised before the hierarchy existed.",
        )?;
        make_type(py, "FerricError", &[py.get_type::<pyo3::exceptions::PyException>().into_any()], &ns)
    })?;
    Ok(t.bind(py))
}

fn make_type<'py>(
    py: Python<'py>,
    name: &str,
    bases: &[Bound<'py, PyAny>],
    ns: &Bound<'py, PyDict>,
) -> PyResult<Py<PyType>> {
    let bases = PyTuple::new(py, bases)?;
    let ty = py.get_type::<PyType>().call1((name, bases, ns))?;
    Ok(ty.downcast_into::<PyType>()?.unbind())
}

fn classes(py: Python<'_>) -> PyResult<&[Py<PyType>]> {
    let v = CLASSES.get_or_try_init(py, || -> PyResult<Vec<Py<PyType>>> {
        let root = root(py)?.clone().into_any();
        let builtins = py.import("builtins")?;
        let mut made: Vec<Py<PyType>> = Vec::new();
        for k in Kind::ALL {
            let ns = PyDict::new(py);
            ns.set_item("__module__", "ferric")?;
            ns.set_item("__doc__", k.doc())?;
            let bases = if let Kind::Basis = k {
                let input = made[Kind::Input.idx()].bind(py).clone().into_any();
                [input, builtins.getattr("RuntimeError")?]
            } else {
                [root.clone(), builtins.getattr(k.builtin())?]
            };
            made.push(make_type(py, k.name(), &bases, &ns)?);
        }
        Ok(made)
    })?;
    Ok(v)
}

/// Add `FerricError` and every subclass to the `ferric` module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("FerricError", root(py)?)?;
    for (k, ty) in Kind::ALL.iter().zip(classes(py)?) {
        m.add(k.name(), ty.bind(py))?;
    }
    Ok(())
}

/// Raise `kind` with `msg` unchanged. Takes the GIL eagerly: every caller is
/// already on a thread that may hold or acquire it, and the class table is
/// built once under it.
pub(crate) fn raise(kind: Kind, msg: String) -> PyErr {
    Python::with_gil(|py| match classes(py) {
        Ok(cls) => PyErr::from_type(cls[kind.idx()].bind(py).clone(), (msg,)),
        // Class creation cannot fail in practice; fall back to the builtin.
        Err(_) => match kind {
            Kind::Unsupported => pyo3::exceptions::PyNotImplementedError::new_err(msg),
            Kind::Input => pyo3::exceptions::PyValueError::new_err(msg),
            _ => pyo3::exceptions::PyRuntimeError::new_err(msg),
        },
    })
}

/// The single `FerricError` -> exception-class decision. The match on
/// [`ErrorCategory`] is exhaustive; `FerricError::category` is exhaustive in
/// ferric-core, so a new variant cannot reach here undecided.
pub(crate) fn kind_of(e: &CoreError) -> Kind {
    match e.category() {
        ErrorCategory::Convergence => Kind::Convergence,
        ErrorCategory::Numerical => Kind::Numerical,
        ErrorCategory::Memory => Kind::MemoryBudget,
        ErrorCategory::Unsupported => Kind::Unsupported,
        ErrorCategory::Basis => Kind::Basis,
        ErrorCategory::Other => Kind::Computation,
    }
}

/// Convert a core error; the message is its `Display` text.
pub(crate) fn from_core(e: &CoreError) -> PyErr {
    raise(kind_of(e), e.to_string())
}

/// [`from_core`] when `e` is a core error, else a `ComputationError` (see [`core_of`]).
pub(crate) fn core_or_computation<E: std::fmt::Display + 'static>(e: E) -> PyErr {
    match core_of(&e) {
        Some(core) => from_core(core),
        None => computation(e),
    }
}

/// Like [`from_core`] for a call site that prefixes the message (`"run_x: {e}"`).
pub(crate) fn from_core_msg(e: &CoreError, msg: String) -> PyErr {
    raise(kind_of(e), msg)
}

/// A failure with no typed core variant (a formatted message): a
/// `ComputationError`, still a `RuntimeError`.
pub(crate) fn computation(msg: impl std::fmt::Display) -> PyErr {
    raise(Kind::Computation, msg.to_string())
}

/// The core error behind `e`, if it is one. `make_err` and `input_err` take
/// any `Display` (the sites pass a dozen foreign error types, strings and
/// core errors), so a core error is recognised by a `'static` downcast; a
/// value borrowed for less than `'static` must be `to_string()`ed first, and
/// then it is untyped. Prefer [`from_core`] / [`input_err_msg`] where the
/// type is known.
fn core_of<E: std::fmt::Display + 'static>(e: &E) -> Option<&CoreError> {
    (e as &dyn std::any::Any).downcast_ref::<CoreError>()
}

/// An invalid argument or option value, raised as a `ValueError` before the
/// hierarchy: `InputError`. A [`CoreError::Basis`] passed here (a bad basis
/// name) is a `BasisError`, which is also a `ValueError`; every other core
/// error is an `InputError` whatever its variant, since these sites always
/// raised `ValueError`.
pub(crate) fn input_err<E: std::fmt::Display + 'static>(e: E) -> PyErr {
    match core_of(&e) {
        Some(core) => input_err_msg(core, e.to_string()),
        None => raise(Kind::Input, e.to_string()),
    }
}

/// [`input_err`] for a call site that prefixes the message: the core error
/// still decides `BasisError` vs `InputError`.
pub(crate) fn input_err_msg(e: &CoreError, msg: String) -> PyErr {
    match e.category() {
        ErrorCategory::Basis => raise(Kind::Basis, msg),
        _ => raise(Kind::Input, msg),
    }
}

/// Raise the Python exception a core error converts to, through the same
/// [`from_core`] every driver uses. Private, for tests: some variants
/// (`Unsupported`, `Lapack`, `Io`) are reachable from Python only through
/// refusals no public keyword triggers, and the mapping table must still be
/// checked class by class. `variant` is a name from [`_core_error_variants`]
/// (built from `FerricError::samples`, so it covers every variant).
#[pyfunction]
pub(crate) fn _raise_core_error(variant: &str, message: &str) -> PyResult<()> {
    match CoreError::samples(message)
        .into_iter()
        .find(|(n, _)| *n == variant)
    {
        Some((_, e)) => Err(from_core(&e)),
        None => Err(input_err(format!("unknown core error variant {variant:?}"))),
    }
}

/// The variant names `_raise_core_error` accepts, one per `FerricError`
/// variant. Private, for tests.
#[pyfunction]
pub(crate) fn _core_error_variants() -> Vec<&'static str> {
    CoreError::samples("").into_iter().map(|(n, _)| n).collect()
}
