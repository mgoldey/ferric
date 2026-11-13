"""QCSchema (MolSSI) atomic input and output for ferric.

``run`` takes a QCSchema ``AtomicInput`` (schema version 1) as a dict and returns
an ``AtomicResult`` dict, or a ``FailedOperation`` dict for any input it cannot
run. It never raises for a bad input:

    import ferric.qcschema as qcs

    out = qcs.run({
        "schema_name": "qcschema_input",
        "schema_version": 1,
        "molecule": {"symbols": ["O", "H", "H"],
                     "geometry": [0, 0, 0.2217, 0, 1.4309, -0.8867, 0, -1.4309, -0.8867]},
        "driver": "gradient",
        "model": {"method": "hf", "basis": "6-31g"},
        "keywords": {},
    })
    out["success"], out["return_result"]       # True, (natom, 3) in Hartree/Bohr

The module needs no QCElemental. ``qcelemental.models.AtomicResult(**out)``
parses the output and ``AtomicInput(...).dict()`` is accepted as input.

Units are QCSchema's: geometry in Bohr (ferric's own unit), energies in
Hartree, gradients in Hartree/Bohr. The geometry is used as given (no
re-orientation), so the gradient is in the input frame. ``geometry`` may be
flat (3N) or (N, 3).

Drivers: ``energy`` and ``gradient``. Any other driver (``hessian``,
``properties``) fails with ``unsupported_error``.

Methods (``model.method``, case-insensitive; ``model.basis`` is a bundled
ferric basis name):

* ``hf`` or ``scf``: RHF for ``molecular_multiplicity`` 1, else UHF
  (``run_rhf`` / ``run_uhf``).
* ``mp2``: RI-MP2 (``run_rimp2``; a UHF reference for multiplicity > 1). The
  auxiliary basis is ``keywords["auxbasis"]`` when given, else
  ``"<basis>-ri"`` (ferric's bundled RI-fitting alias of the orbital basis,
  e.g. ``cc-pvdz`` -> ``cc-pvdz-ri``); an orbital basis with no such alias
  fails with ``input_error`` asking for ``auxbasis``.
* anything else is an XC functional name for closed-shell Kohn-Sham
  (``run_dft``). An unknown functional fails with ``input_error``. An
  open-shell KS request (multiplicity > 1) fails with ``unsupported_error``:
  ferric has no open-shell molecular KS entry point with gradients in Python.

Keywords are a strict allowlist; any other key fails with ``input_error``
naming it. Values are passed to the ferric driver unchanged.

* hf / scf: ``max_iter``, ``energy_conv``, ``density_conv``, ``diis_size``,
  ``integral_thresh``, ``level_shift``, ``k_builder``, ``df_j_aux``,
  ``df_k_aux``, ``ri_j_precision``, ``ri_k_precision``, ``memory_budget_gb``.
* mp2: ``auxbasis``, ``frozen_core``, ``memory_budget_gb``.
* dft: ``max_iter``, ``energy_conv``, ``density_conv``, ``level_shift``,
  ``k_builder``, ``df_j_aux``, ``df_k_aux``, ``ri_j_precision``,
  ``ri_k_precision``, ``memory_budget_gb``, ``grid_radial``, ``grid_angular``,
  ``grid_prune``, ``dispersion``.

A gradient run uses the driver's ``with_gradient=True``, so an unset
``ri_j_precision`` is ``"f64"`` there (see ``run_rhf``); an energy run keeps
the driver default. Pass ``ri_j_precision`` to make both lie on one surface.

Molecule: ``symbols``, ``geometry``, ``molecular_charge`` (default 0),
``molecular_multiplicity`` (default: 1 for an even electron count, else 2) and
``real`` (False marks a ghost atom). Other molecule fields (``masses``,
``fragments``, ``name``, ...) do not change the electronic energy and are
echoed, not used.

Output properties: ``return_energy``, ``return_gradient`` (gradient driver),
``scf_total_energy``, ``scf_iterations`` (SCF and DFT), ``scf_xc_energy``
(DFT), ``mp2_correlation_energy`` and ``mp2_total_energy`` (mp2),
``nuclear_repulsion_energy`` and ``calcinfo_natom`` / ``nalpha`` / ``nbeta``;
``calcinfo_nbasis`` and ``calcinfo_nmo`` for SCF and DFT (the MP2 result does
not expose the basis size). ``extras["ferric"]`` carries ``build`` (commit and
dirty flag baked into the extension) and the validation ``certificate``.

``protocols.wavefunction`` other than ``"none"`` fails with
``unsupported_error``; ferric does not serialise a wavefunction here.

``error.error_type`` is one of ``input_error`` (malformed input, unknown
keyword, method, functional or basis, driver refusal of a keyword),
``unsupported_error`` (a valid request ferric does not provide),
``convergence_error`` (the SCF did not converge) and ``execution_error``
(any other failure while running).
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np

import ferric

__all__ = ["run", "SCHEMA_VERSION"]

SCHEMA_VERSION = 1

_SCF_KEYS = frozenset(
    {
        "max_iter",
        "energy_conv",
        "density_conv",
        "level_shift",
        "k_builder",
        "df_j_aux",
        "df_k_aux",
        "ri_j_precision",
        "ri_k_precision",
        "memory_budget_gb",
    }
)
_KEYWORDS = {
    "hf": _SCF_KEYS | {"diis_size", "integral_thresh"},
    "mp2": frozenset({"auxbasis", "frozen_core", "memory_budget_gb"}),
    "dft": _SCF_KEYS | {"grid_radial", "grid_angular", "grid_prune", "dispersion"},
}


class _QcError(Exception):
    """A failure with a QCSchema ``error_type``."""

    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type


def run(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run a QCSchema ``AtomicInput`` dict; return an ``AtomicResult`` or
    ``FailedOperation`` dict. See the module docstring."""
    try:
        return _run(input_data)
    except _QcError as err:
        return _failure(input_data, err.error_type, str(err))
    except ValueError as err:
        return _failure(input_data, "input_error", f"{type(err).__name__}: {err}")
    except RuntimeError as err:
        text = str(err).lower()
        if "converge" in text:
            kind = "convergence_error"
        elif "unknownname" in text:
            kind = "input_error"
        else:
            kind = "execution_error"
        return _failure(input_data, kind, f"{type(err).__name__}: {err}")
    except Exception as err:  # noqa: BLE001 - the contract is "never raise"
        return _failure(input_data, "execution_error", f"{type(err).__name__}: {err}")


def _failure(input_data: Any, error_type: str, message: str) -> dict[str, Any]:
    try:
        echoed = copy.deepcopy(input_data)
    except Exception:  # noqa: BLE001
        echoed = None
    return {
        "success": False,
        "error": {"error_type": error_type, "error_message": message},
        "input_data": echoed,
    }


def _run(inp: Any) -> dict[str, Any]:
    if not isinstance(inp, dict):
        raise _QcError("input_error", "input must be a dict (a QCSchema AtomicInput)")
    if inp.get("schema_name", "qcschema_input") != "qcschema_input":
        raise _QcError(
            "input_error",
            f"schema_name {inp.get('schema_name')!r} is not 'qcschema_input'",
        )
    if inp.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise _QcError(
            "input_error",
            f"schema_version {inp.get('schema_version')!r}: only version 1 is supported",
        )
    driver = inp.get("driver")
    if driver not in ("energy", "gradient"):
        kind = (
            "unsupported_error"
            if driver in ("hessian", "properties")
            else "input_error"
        )
        raise _QcError(
            kind, f"driver {driver!r}: ferric.qcschema supports 'energy' and 'gradient'"
        )
    model = inp.get("model")
    if not isinstance(model, dict) or "method" not in model:
        raise _QcError("input_error", "model must be a dict with 'method' and 'basis'")
    basis_name = model.get("basis")
    if not isinstance(basis_name, str) or not basis_name:
        raise _QcError(
            "input_error", "model.basis must be a bundled ferric basis name (a string)"
        )
    protocols = inp.get("protocols") or {}
    wfn = protocols.get("wavefunction", "none")
    if wfn != "none":
        raise _QcError(
            "unsupported_error",
            f"protocols.wavefunction {wfn!r}: ferric.qcschema returns no wavefunction",
        )
    keywords = dict(inp.get("keywords") or {})

    method = str(model["method"]).lower()
    family = "hf" if method in ("hf", "scf") else "mp2" if method == "mp2" else "dft"
    unknown = sorted(set(keywords) - _KEYWORDS[family])
    if unknown:
        raise _QcError(
            "input_error",
            f"unknown keyword(s) {unknown} for method {model['method']!r}; allowed: "
            f"{sorted(_KEYWORDS[family])}",
        )

    mol, mult = _molecule(inp.get("molecule"))
    basis = _bundled(basis_name, "basis")
    want_grad = driver == "gradient"
    t0 = time.perf_counter()
    result, props, ret = _dispatch(
        family, method, mol, mult, basis, basis_name, keywords, want_grad
    )
    wall = time.perf_counter() - t0

    nelec = mol.nelec()
    props["nuclear_repulsion_energy"] = mol.nuclear_repulsion()
    props["calcinfo_natom"] = mol.natoms()
    props["calcinfo_nalpha"] = (nelec + mult - 1) // 2
    props["calcinfo_nbeta"] = (nelec - mult + 1) // 2
    props["return_energy"] = ret["energy"]
    if want_grad:
        gradient = result.gradient()
        if gradient is None:
            raise _QcError("execution_error", "the ferric result carries no gradient")
        gradient = np.asarray(gradient, dtype=float)
        props["return_gradient"] = gradient.tolist()
        return_result: Any = gradient.tolist()
    else:
        return_result = ret["energy"]

    extras = dict(inp.get("extras") or {})
    extras["ferric"] = {
        "build": dict(ferric.build_info()),
        "certificate": _plain(result.certificate),
    }
    out = copy.deepcopy(inp)
    out.update(
        {
            "schema_name": "qcschema_output",
            "schema_version": SCHEMA_VERSION,
            "return_result": return_result,
            "properties": props,
            "success": True,
            "provenance": {
                "creator": "ferric",
                "version": ferric.__version__,
                "routine": f"ferric.qcschema.run:{family}",
                "wall_time": wall,
            },
            "extras": extras,
            "stdout": None,
            "stderr": None,
        }
    )
    return out


def _dispatch(family, method, mol, mult, basis, basis_name, keywords, want_grad):
    kw = dict(keywords)
    if want_grad:
        kw["with_gradient"] = True
    props: dict[str, Any] = {}
    if family == "hf":
        driver = ferric.run_rhf if mult == 1 else ferric.run_uhf
        result = driver(mol, basis, **kw)
        _require_converged(result, driver.__name__)
        _scf_props(props, result, mult)
        props["scf_total_energy"] = result.energy
        return result, props, {"energy": result.energy}
    if family == "mp2":
        aux_name = kw.pop("auxbasis", None)
        if aux_name is None:
            aux_name = f"{basis_name}-ri"
            try:
                aux = ferric.BasisSet.bundled(aux_name)
            except ValueError as err:
                raise _QcError(
                    "input_error",
                    f"no bundled RI basis {aux_name!r} for {basis_name!r}; pass keywords['auxbasis']",
                ) from err
        else:
            aux = _bundled(aux_name, "auxbasis")
        result = ferric.run_rimp2(mol, basis, aux, **kw)
        props["scf_total_energy"] = result.rhf_energy
        props["mp2_correlation_energy"] = result.mp2_corr
        props["mp2_total_energy"] = result.total_energy
        return result, props, {"energy": result.total_energy}
    if mult != 1:
        raise _QcError(
            "unsupported_error",
            f"method {method!r} with multiplicity {mult}: open-shell molecular Kohn-Sham is not "
            "available (run_dft is closed-shell)",
        )
    result = ferric.run_dft(mol, basis, functional=method, **kw)
    _require_converged(result, "run_dft")
    _scf_props(props, result, mult)
    props["scf_total_energy"] = result.total_energy
    props["scf_xc_energy"] = result.e_xc
    return result, props, {"energy": result.total_energy}


def _require_converged(result, what):
    if not result.converged:
        raise _QcError("convergence_error", f"{what}: SCF did not converge")


def _scf_props(props, result, mult):
    if hasattr(result, "iterations"):
        props["scf_iterations"] = int(result.iterations)
    if hasattr(result, "mo_coefficients"):
        mo = np.asarray(result.mo_coefficients())
    else:
        mo = np.asarray(result.mo_coefficients_alpha())
    props["calcinfo_nbasis"] = int(mo.shape[0])
    props["calcinfo_nmo"] = int(mo.shape[1])


def _bundled(name, what):
    try:
        return ferric.BasisSet.bundled(name)
    except ValueError as err:
        raise _QcError("input_error", f"{what} {name!r}: {err}") from err


def _molecule(mol):
    if not isinstance(mol, dict):
        raise _QcError("input_error", "molecule must be a dict")
    symbols, geometry = mol.get("symbols"), mol.get("geometry")
    if symbols is None or len(symbols) == 0 or geometry is None:
        raise _QcError("input_error", "molecule needs 'symbols' and 'geometry'")
    coords = np.asarray(geometry, dtype=float)
    if coords.size != 3 * len(symbols):
        raise _QcError(
            "input_error",
            f"geometry has {coords.size} numbers for {len(symbols)} atoms (need 3 per atom)",
        )
    coords = coords.reshape(len(symbols), 3)
    ghost = None
    if mol.get("real") is not None:
        ghost = [not bool(r) for r in mol["real"]]
    common = dict(units="bohr", charge=int(mol.get("molecular_charge", 0)), ghost=ghost)
    if mol.get("molecular_multiplicity") is not None:
        mult = int(mol["molecular_multiplicity"])
        return ferric.Molecule.from_arrays(
            list(symbols), coords, multiplicity=mult, **common
        ), mult
    try:
        return ferric.Molecule.from_arrays(
            list(symbols), coords, multiplicity=1, **common
        ), 1
    except ValueError:
        return ferric.Molecule.from_arrays(
            list(symbols), coords, multiplicity=2, **common
        ), 2


def _plain(obj):
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
