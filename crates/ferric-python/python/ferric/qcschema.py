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

Drivers: ``energy``, ``gradient`` and ``hessian``. Any other driver
(``properties``) fails with ``unsupported_error``. ``hessian`` is the analytic
Hessian (``run_frequencies(hessian="analytic")`` /
``"analytic-moving-grid"``, never a finite difference) for ``hf`` / ``scf``
(RHF for a singlet, UHF otherwise) and for DFT (RKS for a singlet, UKS
otherwise); ``return_result`` is the (3N, 3N) Cartesian Hessian in
Hartree/Bohr^2 in the input frame. Any method or setting without an analytic
Hessian (mp2, ccsd, ROKS, an ECP, a range-separated or meta-GGA functional
for UKS...) fails with ``unsupported_error``.

Methods (``model.method``, case-insensitive; ``model.basis`` is a bundled
ferric basis name):

* ``hf`` or ``scf``: RHF for ``molecular_multiplicity`` 1, else UHF
  (``run_rhf`` / ``run_uhf``).
* ``mp2``: RI-MP2 (``run_rimp2``; a UHF reference for multiplicity > 1). The
  auxiliary basis is ``keywords["auxbasis"]`` when given, else
  ``"<basis>-ri"`` (ferric's bundled RI-fitting alias of the orbital basis,
  e.g. ``cc-pvdz`` -> ``cc-pvdz-ri``); an orbital basis with no such alias
  fails with ``input_error`` asking for ``auxbasis``.
* ``ccsd`` and ``ccsd(t)``: closed-shell spin-adapted CC on an RHF reference
  (``run_ccsd`` / ``run_ccsd_t``), auxiliary basis as for mp2. Energy driver
  only: a gradient or Hessian request, or multiplicity > 1, fails with
  ``unsupported_error``.
* anything else is an XC functional name for Kohn-Sham: RKS for
  ``molecular_multiplicity`` 1 (``run_dft``), UKS otherwise (``run_uks``;
  energy and gradient). An unknown functional fails with ``input_error``.
  ``dispersion`` is closed-shell only: with multiplicity > 1 it fails with
  ``unsupported_error``.

Keywords are a strict allowlist; any other key fails with ``input_error``
naming it. Values are passed to the ferric driver unchanged.

* hf / scf: ``max_iter``, ``energy_conv``, ``density_conv``, ``diis_size``,
  ``integral_thresh``, ``level_shift``, ``k_builder``, ``df_j_aux``,
  ``df_k_aux``, ``ri_j_precision``, ``ri_k_precision``, ``memory_budget_gb``.
* mp2: ``auxbasis``, ``frozen_core``, ``memory_budget_gb``.
* ccsd / ccsd(t): ``auxbasis``, ``frozen_core``, ``memory_budget_gb``,
  ``k_builder``, ``ladder``, ``scf_max_iter``, ``scf_energy_conv``,
  ``scf_density_conv``.
* hessian driver (hf and dft): ``max_iter``, ``energy_conv``, ``density_conv``,
  ``df_j_aux``, ``df_k_aux``.
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
``ccsd_correlation_energy`` / ``ccsd_total_energy`` (ccsd and ccsd(t)),
``ccsd_prt_pr_correlation_energy`` / ``ccsd_prt_pr_total_energy`` (ccsd(t)),
``return_hessian`` (hessian driver),
``nuclear_repulsion_energy`` and ``calcinfo_natom`` / ``nalpha`` / ``nbeta``;
``calcinfo_nbasis`` and ``calcinfo_nmo`` for SCF and DFT (the MP2 result does
not expose the basis size). ``provenance["build"]`` is ``ferric.build_info()``
(commit and dirty flag baked into the extension) and
``extras["ferric"]["certificate"]`` is the result's validation certificate.

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

from ._util import _plain, default_multiplicity

__all__ = [
    "SCHEMA_VERSION",
    "molecule_from_qcschema",
    "molecule_to_qcschema",
    "run",
]

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
    "cc": frozenset(
        {
            "auxbasis",
            "frozen_core",
            "memory_budget_gb",
            "k_builder",
            "ladder",
            "scf_max_iter",
            "scf_energy_conv",
            "scf_density_conv",
        }
    ),
}
_HESSIAN_KEYS = frozenset(
    {"max_iter", "energy_conv", "density_conv", "df_j_aux", "df_k_aux"}
)
_CC_METHODS = ("ccsd", "ccsd(t)")


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
    if driver not in ("energy", "gradient", "hessian"):
        kind = "unsupported_error" if driver == "properties" else "input_error"
        raise _QcError(
            kind,
            f"driver {driver!r}: ferric.qcschema supports 'energy', 'gradient' and 'hessian'",
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
    family = _family(method)
    allowed = _HESSIAN_KEYS if driver == "hessian" else _KEYWORDS[family]
    unknown = sorted(set(keywords) - allowed)
    if unknown:
        raise _QcError(
            "input_error",
            f"unknown keyword(s) {unknown} for method {model['method']!r}"
            f" with driver {driver!r}; allowed: {sorted(allowed)}",
        )

    mol, mult = _molecule(inp.get("molecule"))
    basis = _bundled(basis_name, "basis")
    want_grad = driver == "gradient"
    t0 = time.perf_counter()
    if driver == "hessian":
        result, props, ret = _hessian(family, method, mol, mult, basis_name, keywords)
    else:
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
    elif driver == "hessian":
        return_result = ret["hessian"]
        props["return_hessian"] = return_result
    else:
        return_result = ret["energy"]

    extras = dict(inp.get("extras") or {})
    extras["ferric"] = {"certificate": _plain(result.certificate)}
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
                "build": dict(ferric.build_info()),
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
            aux = _ri_basis(basis_name)
        else:
            aux = _bundled(aux_name, "auxbasis")
        result = ferric.run_rimp2(mol, basis, aux, **kw)
        props["scf_total_energy"] = result.rhf_energy
        props["mp2_correlation_energy"] = result.mp2_corr
        props["mp2_total_energy"] = result.total_energy
        return result, props, {"energy": result.total_energy}
    if family == "cc":
        return _cc(method, mol, mult, basis, basis_name, kw, want_grad)
    if mult != 1:
        return _open_shell_dft(method, mol, mult, basis, kw)
    result = ferric.run_dft(mol, basis, functional=method, **kw)
    _require_converged(result, "run_dft")
    _scf_props(props, result, mult)
    props["scf_total_energy"] = result.total_energy
    props["scf_xc_energy"] = result.e_xc
    return result, props, {"energy": result.total_energy}


def _open_shell_dft(method, mol, mult, basis, kw):
    """UKS (``run_uks``): energy and, with ``with_gradient``, the analytic gradient."""
    if "dispersion" in kw:
        raise _QcError(
            "unsupported_error",
            f"method {method!r} with multiplicity {mult}: dispersion is closed-shell only "
            "(run_uks has no dispersion)",
        )
    result = ferric.run_uks(mol, basis, method, **kw)
    _require_converged(result, "run_uks")
    props: dict[str, Any] = {}
    _scf_props(props, result, mult)
    props["scf_total_energy"] = result.energy
    return result, props, {"energy": result.energy}


def _family(method):
    if method in ("hf", "scf"):
        return "hf"
    if method == "mp2":
        return "mp2"
    return "cc" if method in _CC_METHODS else "dft"


def _cc(method, mol, mult, basis, basis_name, kw, want_grad):
    """Closed-shell CCSD / CCSD(T): the correlation energy is the driver's; the
    reference energy is a separate ``run_rhf`` call with the same SCF controls
    (``CcResult`` does not expose its internal reference)."""
    if want_grad:
        raise _QcError(
            "unsupported_error", f"method {method!r}: ferric has no CC gradient"
        )
    if mult != 1:
        raise _QcError(
            "unsupported_error",
            f"method {method!r} with multiplicity {mult}: ferric CC is closed-shell (RHF reference)",
        )
    aux_name = kw.pop("auxbasis", None)
    if aux_name is None:
        aux = _ri_basis(basis_name)
    else:
        aux = _bundled(aux_name, "auxbasis")
    ref_kw = {"max_iter": "scf_max_iter", "energy_conv": "scf_energy_conv"}
    ref_kw = {k: kw[v] for k, v in ref_kw.items() if v in kw}
    if "scf_density_conv" in kw:
        ref_kw["density_conv"] = kw["scf_density_conv"]
    if "k_builder" in kw:
        ref_kw["k_builder"] = kw["k_builder"]
    if "memory_budget_gb" in kw:
        ref_kw["memory_budget_gb"] = kw["memory_budget_gb"]
    driver = ferric.run_ccsd_t if method == "ccsd(t)" else ferric.run_ccsd
    result = driver(mol, basis, aux, **kw)
    reference = ferric.run_rhf(mol, basis, **ref_kw)
    _require_converged(reference, "run_rhf")
    props: dict[str, Any] = {"scf_total_energy": reference.energy}
    _scf_props(props, reference, mult)
    e_ccsd_corr = result.correlation_energy
    props["ccsd_correlation_energy"] = e_ccsd_corr
    props["ccsd_total_energy"] = reference.energy + e_ccsd_corr
    energy = props["ccsd_total_energy"]
    if method == "ccsd(t)":
        e_corr = e_ccsd_corr + result.t_correction
        props["ccsd_prt_pr_correlation_energy"] = e_corr
        props["ccsd_prt_pr_total_energy"] = reference.energy + e_corr
        energy = props["ccsd_prt_pr_total_energy"]
    return result, props, {"energy": energy}


def _hessian(family, method, mol, mult, basis_name, keywords):
    """Analytic Hessian through ``run_frequencies``; ``hessian="analytic"`` /
    ``"analytic-moving-grid"`` never falls back to finite differences."""
    # run_frequencies spells the SCF controls scf_<name> (the bare names warn).
    kw = {
        (f"scf_{k}" if k in ("max_iter", "energy_conv", "density_conv") else k): v
        for k, v in keywords.items()
    }
    if family == "hf":
        kw.update(reference="rhf" if mult == 1 else "uhf", hessian="analytic")
    elif family == "dft":
        kw.update(
            xc=method,
            reference="rhf" if mult == 1 else "uhf",
            hessian="analytic-moving-grid",
        )
    else:
        raise _QcError(
            "unsupported_error",
            f"method {method!r} with multiplicity {mult}: ferric has no analytic Hessian",
        )
    try:
        result = ferric.run_frequencies(mol, basis_name, **kw)
    except (ValueError, RuntimeError) as err:
        # The driver's typed refusals (a setting the analytic Hessian does not
        # cover) name "analytic" and "Hessian"; anything else is not one.
        text = str(err).lower()
        if "analytic" in text and "hessian" in text:
            raise _QcError(
                "unsupported_error", f"no analytic Hessian here: {err}"
            ) from err
        raise
    props: dict[str, Any] = {"scf_total_energy": result.energy}
    hessian = np.asarray(result.cartesian_hessian, dtype=float)
    return result, props, {"energy": result.energy, "hessian": hessian.tolist()}


def _ri_basis(basis_name):
    aux_name = f"{basis_name}-ri"
    try:
        return ferric.BasisSet.bundled(aux_name)
    except ValueError as err:
        raise _QcError(
            "input_error",
            f"no bundled RI basis {aux_name!r} for {basis_name!r}; pass keywords['auxbasis']",
        ) from err


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


def molecule_from_qcschema(d: dict[str, Any]) -> ferric.Molecule:
    """A ``ferric.Molecule`` from a QCSchema molecule dict.

    Reads ``symbols``, ``geometry`` (Bohr, flat or (N, 3)),
    ``molecular_charge``, ``molecular_multiplicity`` (default: 1 for an even
    electron count, else 2) and ``real`` (False marks a ghost atom). Raises
    ``ValueError`` for a malformed molecule.
    """
    try:
        return _molecule(d)[0]
    except _QcError as err:
        raise ValueError(str(err)) from err


def molecule_to_qcschema(
    mol: ferric.Molecule, charge: int | None = None, multiplicity: int | None = None
) -> dict[str, Any]:
    """A QCSchema molecule dict for a ``ferric.Molecule``.

    ``geometry`` is flat (3N) in Bohr, ``real`` carries the ghost flags.
    ``charge`` and ``multiplicity`` default to the Molecule's own; explicit
    values override. ``molecule_from_qcschema`` reads them back. The
    coordinates are ferric's own Bohr numbers, so the round trip is exact.
    """
    coords = np.asarray(mol.coords_bohr(), dtype=float)
    charge = mol.charge if charge is None else charge
    multiplicity = mol.multiplicity if multiplicity is None else multiplicity
    return {
        "schema_name": "qcschema_molecule",
        "schema_version": 2,
        "symbols": list(mol.symbols()),
        "geometry": coords.reshape(-1).tolist(),
        "real": [not g for g in mol.is_ghost()],
        "molecular_charge": int(charge),
        "molecular_multiplicity": int(multiplicity),
    }


def _molecule(mol):
    """Private form of ``molecule_from_qcschema``: ``(Molecule, multiplicity)``,
    raising ``_QcError`` (``input_error``) on a bad molecule."""
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
    mult = default_multiplicity(symbols, common["charge"], ghost)
    return ferric.Molecule.from_arrays(
        list(symbols), coords, multiplicity=mult, **common
    ), mult
