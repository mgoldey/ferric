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

Drivers: ``energy``, ``gradient``, ``hessian`` and ``properties``. Any other
driver fails with ``input_error``. ``hessian`` is the analytic
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

``properties`` returns a dict as ``return_result`` (QCSchema's convention for
that driver), atomic units throughout. ``keywords["properties"]`` lists what to
compute; without it the defaults are returned (SCF and Kohn-Sham only; MP2 and
CC return their energies). A name that cannot be computed fails the whole job:

* ``scf_dipole_moment`` (3, e*Bohr, about the coordinate origin) and
  ``scf_quadrupole_moment`` (3x3, e*Bohr^2, raw Cartesian second moment about
  the origin, not traceless; closed shell only): the SCF/KS density, also copied
  into ``properties``. ECP cores and ghost atoms are handled (effective charges).
* ``orbital_energies_alpha`` / ``orbital_energies_beta`` (Hartree; beta only for
  an unrestricted reference), ``homo_energy``, ``lumo_energy`` (highest
  occupied / lowest virtual over both spins).
* ``mulliken_charges``, ``lowdin_charges`` (closed shell only) and, on request
  only, ``hirshfeld_charges`` and ``chelpg_charges`` (closed shell only), in e.
* Energies already in ``properties`` (``scf_total_energy``...) may be named.
* ``mp2_dipole_moment``, ``ccsd_dipole_moment``... and every density property
  with ``mp2`` / ``ccsd`` / ``ccsd(t)``: ``unsupported_error``, ferric's MP2 and
  CC results expose no density. An unknown name is ``input_error``.

The names without a QCSchema ``AtomicResultProperties`` field (orbital
energies, HOMO/LUMO, charges) appear only in ``return_result``: the schema
forbids extra fields in ``properties``.

Keywords are a strict allowlist (plus ``properties`` for that driver); any other key fails with ``input_error``
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
``calcinfo_nbasis`` for every method and driver (``BasisSet.n_functions``) and
``calcinfo_nmo`` for SCF, DFT and CC (the MP2 result does not expose the MO
count). ``provenance["build"]`` is ``ferric.build_info()``
(commit and dirty flag baked into the extension) and
``extras["ferric"]["certificate"]`` is the result's validation certificate.

``protocols.wavefunction`` other than ``"none"`` fails with
``unsupported_error``: QCSchema v1 orbitals carry no AO ordering or
normalisation convention, and ferric's (libint: spherical m = -l..l, Cartesian
components sharing the x^l norm) differs from other producers', so a consumer
would read the coefficients wrongly without any error. ``protocols.native_files``
other than ``"none"`` fails the same way. Orbital energies are available
through the ``properties`` driver instead.

``error.error_type`` is one of ``input_error`` (malformed input, unknown
keyword, method, functional or basis, driver refusal of a keyword),
``unsupported_error`` (a valid request ferric does not provide),
``convergence_error`` (the SCF did not converge) and ``execution_error``
(any other failure while running).

Failures are classified by ferric's exception classes: ``ferric.UnsupportedError``
is ``unsupported_error``, ``ferric.ConvergenceError`` is ``convergence_error``,
``ferric.InputError`` and ``ferric.BasisError`` (also a ``ValueError``) are
``input_error``, and a plain ``RuntimeError`` or ``ferric.ComputationError`` falls
back to sniffing its text. Since the hierarchy was added, a basis failure raised
inside a driver (an unknown auxiliary basis, say) is an ``input_error`` (it was
``execution_error``) and a refusal a driver raises as ``UnsupportedError`` is an
``unsupported_error``; the ``error_message`` prefix is the new class name
(``BasisError: ...``, not ``RuntimeError: ...``).
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
    except ferric.UnsupportedError as err:
        return _failure(input_data, "unsupported_error", f"{type(err).__name__}: {err}")
    except ferric.ConvergenceError as err:
        return _failure(input_data, "convergence_error", f"{type(err).__name__}: {err}")
    except ValueError as err:  # InputError, BasisError and plain ValueError
        return _failure(input_data, "input_error", f"{type(err).__name__}: {err}")
    except RuntimeError as err:
        # Untyped failures (ComputationError and plain RuntimeError): sniff.
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
    if driver not in ("energy", "gradient", "hessian", "properties"):
        raise _QcError(
            "input_error",
            f"driver {driver!r}: ferric.qcschema supports 'energy', 'gradient',"
            " 'hessian' and 'properties'",
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
            f"protocols.wavefunction {wfn!r}: ferric.qcschema returns no wavefunction"
            " (QCSchema v1 carries no AO ordering or normalisation, and ferric's"
            " libint AO order would be misread by consumers; see the module docstring)",
        )
    native = protocols.get("native_files", "none")
    if native != "none":
        raise _QcError(
            "unsupported_error",
            f"protocols.native_files {native!r}: ferric.qcschema returns no native files",
        )
    keywords = dict(inp.get("keywords") or {})
    requested = None
    if driver == "properties":
        requested = _requested_properties(keywords.pop("properties", None))

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
    _check_properties_supported(requested, family, method)
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
    try:
        props.setdefault("calcinfo_nbasis", int(basis.n_functions(mol)))
    except ValueError:  # an element the basis does not cover: the run would have failed
        pass
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
    elif driver == "properties":
        return_result = _properties(family, mol, mult, basis, result, props, requested)
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


# Properties of the ``properties`` driver. SCHEMA names are fields of
# QCSchema's AtomicResultProperties (so they also land in ``properties``); the
# rest have no schema field and live in ``return_result`` only.
_SCHEMA_PROPERTIES = ("scf_dipole_moment", "scf_quadrupole_moment")
_EXTENSION_PROPERTIES = (
    "orbital_energies_alpha",
    "orbital_energies_beta",
    "homo_energy",
    "lumo_energy",
    "mulliken_charges",
    "lowdin_charges",
    "hirshfeld_charges",
    "chelpg_charges",
)
_SUPPORTED_PROPERTIES = _SCHEMA_PROPERTIES + _EXTENSION_PROPERTIES
# Computed on request only: both run extra SCF or grid work.
_OPT_IN = ("hirshfeld_charges", "chelpg_charges")
_CLOSED_SHELL_ONLY = ("mulliken_charges", "lowdin_charges") + _OPT_IN
# Schema fields that need a post-SCF density ferric does not expose.
_NO_DENSITY = (
    "mp2_dipole_moment",
    "ccsd_dipole_moment",
    "ccsd_prt_pr_dipole_moment",
    "ccsdt_dipole_moment",
    "ccsdtq_dipole_moment",
)
# Energy names are always in ``properties``; asking for one is not an error
# when the method produced it.
_ENERGY_NAMES = frozenset(
    {
        "return_energy",
        "scf_total_energy",
        "scf_xc_energy",
        "nuclear_repulsion_energy",
        "mp2_correlation_energy",
        "mp2_total_energy",
        "ccsd_correlation_energy",
        "ccsd_total_energy",
        "ccsd_prt_pr_correlation_energy",
        "ccsd_prt_pr_total_energy",
    }
)


def _requested_properties(value):
    """The ``keywords["properties"]`` list (``None`` when absent), validated."""
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise _QcError("input_error", "keywords['properties'] must be a list of names")
    names = list(dict.fromkeys(value))  # drop repeats, keep the order
    for name in names:
        if name in _SUPPORTED_PROPERTIES or name in _ENERGY_NAMES:
            continue
        if name in _NO_DENSITY:
            continue  # refused by method in _check_properties_supported
        raise _QcError(
            "input_error",
            f"unknown property {name!r}; supported: {list(_SUPPORTED_PROPERTIES)}"
            f" and the energies {sorted(_ENERGY_NAMES)}",
        )
    return names


def _check_properties_supported(requested, family, method):
    """Refuse, before any computation, a property the method cannot give."""
    if not requested:
        return
    for name in requested:
        if name in _NO_DENSITY or (
            name in _SUPPORTED_PROPERTIES and family not in ("hf", "dft")
        ):
            raise _QcError(
                "unsupported_error",
                f"property {name!r} with method {method!r}: ferric's MP2 and CC"
                " results expose no density (relaxed or unrelaxed), so only SCF and"
                " Kohn-Sham properties are available",
            )


def _scf_arrays(result):
    """(restricted, orbital energies alpha, beta or None, density total)."""
    if hasattr(result, "orbital_energies_alpha"):
        density = np.asarray(result.density_alpha()) + np.asarray(result.density_beta())
        return (
            False,
            np.asarray(result.orbital_energies_alpha(), dtype=float),
            np.asarray(result.orbital_energies_beta(), dtype=float),
            density,
        )
    return (
        True,
        np.asarray(result.orbital_energies(), dtype=float),
        None,
        np.asarray(result.density()),
    )


def _properties(family, mol, mult, basis, result, props, requested):
    """The ``properties`` driver: the dict returned as ``return_result``.

    Atomic units throughout (QCSchema): dipole in e*Bohr and quadrupole in
    e*Bohr^2 about the coordinate origin (the quadrupole is the raw Cartesian
    second moment sum_A Z_A R_A R_A - sum_uv D_uv <u|r r|v>, not traceless),
    charges in e, orbital energies in Hartree. Schema-named entries are also
    written to ``props``. Without ``keywords["properties"]`` every default
    property the reference supports is returned; a requested property that is
    not computable raises (nothing is skipped silently).
    """
    wanted = list(requested) if requested else None
    out: dict[str, Any] = {}
    if family not in ("hf", "dft"):
        # MP2 / CC: the energies the run produced.
        for name in wanted or [n for n in props if n in _ENERGY_NAMES]:
            out[name] = _energy_property(name, props)
        return out
    restricted, eps_a, eps_b, density = _scf_arrays(result)
    nelec = mol.nelec()
    nalpha, nbeta = (nelec + mult - 1) // 2, (nelec - mult + 1) // 2
    if wanted is None:
        wanted = [
            "scf_dipole_moment",
            *(["scf_quadrupole_moment"] if restricted else []),
            "orbital_energies_alpha",
            *([] if restricted else ["orbital_energies_beta"]),
            "homo_energy",
        ]
        if len(eps_a) > nalpha or (eps_b is not None and len(eps_b) > nbeta):
            wanted.append("lumo_energy")
        if restricted:
            wanted += ["mulliken_charges", "lowdin_charges"]
    for name in wanted:
        if name in out:
            continue
        if name in _ENERGY_NAMES:
            out[name] = _energy_property(name, props)
        elif name == "scf_dipole_moment":
            out[name] = _dipole(mol, basis, density).tolist()
        elif name == "scf_quadrupole_moment":
            if not restricted:
                raise _QcError(
                    "unsupported_error",
                    "scf_quadrupole_moment: ferric's second-moment routine takes an"
                    f" RHF or RKS result only (multiplicity {mult} runs an unrestricted"
                    " reference)",
                )
            out[name] = _quadrupole(mol, basis, result, density).tolist()
        elif name == "orbital_energies_alpha":
            out[name] = eps_a.tolist()
        elif name == "orbital_energies_beta":
            if restricted:
                raise _QcError(
                    "unsupported_error",
                    "orbital_energies_beta: the reference is restricted, beta equals"
                    " orbital_energies_alpha",
                )
            out[name] = eps_b.tolist()
        elif name in ("homo_energy", "lumo_energy"):
            out[name] = _frontier(name, eps_a, eps_b, nalpha, nbeta)
        elif name in _CLOSED_SHELL_ONLY:
            if not restricted:
                raise _QcError(
                    "unsupported_error",
                    f"{name}: ferric's population analyses are closed-shell only"
                    f" (multiplicity {mult} runs an unrestricted reference)",
                )
            fn = getattr(ferric, name)
            out[name] = np.asarray(fn(mol, basis, result), dtype=float).tolist()
    for name in _SCHEMA_PROPERTIES:
        if name in out:
            props[name] = out[name]
    return out


def _energy_property(name, props):
    if name not in props:
        raise _QcError(
            "unsupported_error", f"property {name!r} is not produced by this method"
        )
    return props[name]


def _frontier(name, eps_a, eps_b, nalpha, nbeta):
    """HOMO / LUMO energy: highest occupied / lowest virtual over both spins."""
    homo = [
        e[n - 1]
        for e, n in ((eps_a, nalpha), (eps_b, nbeta))
        if e is not None and n > 0
    ]
    lumo = [
        e[n]
        for e, n in ((eps_a, nalpha), (eps_b, nbeta))
        if e is not None and n < len(e)
    ]
    values = homo if name == "homo_energy" else lumo
    if not values:
        what = "occupied" if name == "homo_energy" else "virtual"
        raise _QcError(
            "unsupported_error", f"{name}: the reference has no {what} orbital"
        )
    return float(max(values) if name == "homo_energy" else min(values))


def _nuclear_charges(mol, basis):
    """Effective nuclear charges: Z minus ECP core electrons, 0 for a ghost."""
    return np.array(
        [
            0.0 if ghost else float(z - basis.ecp_core_electrons(z))
            for z, ghost in zip(mol.atomic_numbers(), mol.is_ghost())
        ]
    )


def _dipole(mol, basis, density):
    ints = ferric.AoIntegrals(mol, basis)
    electronic = -np.einsum("xij,ij->x", np.asarray(ints.dipole()), density)
    return electronic + _nuclear_charges(mol, basis) @ np.asarray(mol.coords_bohr())


def _quadrupole(mol, basis, result, density):
    second = np.asarray(ferric.density_second_moment(mol, basis, result), dtype=float)
    coords = np.asarray(mol.coords_bohr())
    nuclear = np.einsum("a,ai,aj->ij", _nuclear_charges(mol, basis), coords, coords)
    return nuclear - second


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
