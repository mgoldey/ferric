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

use ferric_core::FerricError as CoreError;

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
                let input = made[Kind::Input as usize].bind(py).clone().into_any();
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

/// Raise `kind` with `msg` unchanged.
pub(crate) fn raise(kind: Kind, msg: String) -> PyErr {
    Python::with_gil(|py| match classes(py) {
        Ok(cls) => PyErr::from_type(cls[kind as usize].bind(py).clone(), (msg,)),
        // Class creation cannot fail in practice; fall back to the builtin.
        Err(_) => match kind {
            Kind::Unsupported => pyo3::exceptions::PyNotImplementedError::new_err(msg),
            Kind::Input => pyo3::exceptions::PyValueError::new_err(msg),
            _ => pyo3::exceptions::PyRuntimeError::new_err(msg),
        },
    })
}

/// The single `FerricError` -> exception-class decision.
pub(crate) fn kind_of(e: &CoreError) -> Kind {
    match e {
        CoreError::ScfConvergence { .. } | CoreError::Convergence(_) => Kind::Convergence,
        CoreError::Basis(_) => Kind::Basis,
        CoreError::Lapack(_) => Kind::Numerical,
        CoreError::MemoryExhausted(_) => Kind::MemoryBudget,
        CoreError::Unsupported { .. } => Kind::Unsupported,
        _ => Kind::Computation,
    }
}

/// Convert a core error; the message is its `Display` text.
pub(crate) fn from_core(e: &CoreError) -> PyErr {
    raise(kind_of(e), e.to_string())
}

/// Like [`from_core`] for a call site that prefixes the message (`"run_x: {e}"`).
pub(crate) fn from_core_msg(e: &CoreError, msg: String) -> PyErr {
    raise(kind_of(e), msg)
}

/// An invalid argument or option value, raised as a `ValueError` before the
/// hierarchy: `InputError`. A [`CoreError::Basis`] passed here (a bad basis
/// name) is a `BasisError`, which is also a `ValueError`; every other core
/// error is an `InputError` whatever its variant, since these sites always
/// raised `ValueError`.
pub(crate) fn input_err<E: std::fmt::Display + 'static>(e: E) -> PyErr {
    match (&e as &dyn std::any::Any).downcast_ref::<CoreError>() {
        Some(CoreError::Basis(_)) => raise(Kind::Basis, e.to_string()),
        _ => raise(Kind::Input, e.to_string()),
    }
}

/// Raise the Python exception `FerricError::<variant>` converts to, through
/// the same [`from_core`] every driver uses. Private, for tests: some
/// variants (`Unsupported`, `Lapack`, `Io`) have no reachable Python trigger
/// today, and the mapping table must still be checked class by class.
/// `variant` is one of `io`, `xyz_parse`, `basis`, `libint`, `scf_convergence`,
/// `lapack`, `convergence`, `general`, `memory_exhausted`, `unsupported`.
#[pyfunction]
pub(crate) fn _raise_core_error(variant: &str, message: &str) -> PyResult<()> {
    let m = message.to_string();
    let e = match variant {
        "io" => CoreError::Io(std::io::Error::other(m)),
        "xyz_parse" => CoreError::XyzParse(m),
        "basis" => CoreError::Basis(m),
        "libint" => CoreError::Libint(m),
        "scf_convergence" => CoreError::ScfConvergence {
            iterations: 7,
            last_energy: -1.5,
        },
        "lapack" => CoreError::Lapack(m),
        "convergence" => CoreError::Convergence(m),
        "general" => CoreError::General(m),
        "memory_exhausted" => CoreError::MemoryExhausted(m),
        "unsupported" => CoreError::unsupported("a feature", m),
        other => return Err(input_err(format!("unknown core error variant {other:?}"))),
    };
    Err(from_core(&e))
}
