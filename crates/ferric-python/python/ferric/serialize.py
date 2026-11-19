"""Plain-data views of ferric objects: dicts, JSON, DataFrame-ready records.

    import ferric, ferric.serialize as fs

    res = ferric.run_rhf(mol, basis)
    fs.to_dict(res)                 # {"type": "RhfResult", "energy": ..., ...}
    fs.to_json(res, indent=2)       # non-finite floats become null
    pandas.DataFrame([fs.to_dict(r) for r in results])

A result becomes its public scalar attributes (``int``, ``float``, ``bool``,
``str``, ``None``; read-only getters, not methods), ``"type"`` (the class
name), a normalised ``"energy"`` (the ``energy`` or ``total_energy`` attribute;
the original keys stay), the ``certificate`` when the result has one, and
``"ferric_build"`` (``ferric.__build__``). ``arrays=True`` adds the result's
array accessors (density, MO coefficients, orbital energies, gradient) as
nested lists. A ``ferric.Molecule`` becomes ``symbols``, ``coords_bohr`` and
``ghost``, ``charge`` and ``multiplicity`` (the Molecule's getters). ``molecule_from_dict`` reverses it.

``read_run_log`` parses the CLI's JSON Lines run log (``--json`` /
``<input>.ferric.jsonl``): one object per line, each with ``record`` (kind),
``seq`` and ``t``; the first is ``run_start`` (``schema``, ``ferric_version``,
``git_sha``, ``timestamp``, ``config``, ``molecule``).
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np

import ferric

from ._util import _plain, default_multiplicity

__all__ = ["molecule_from_dict", "read_run_log", "to_dict", "to_json"]

_ARRAY_ACCESSORS = (
    "density",
    "density_alpha",
    "density_beta",
    "mo_coefficients",
    "mo_coefficients_alpha",
    "mo_coefficients_beta",
    "orbital_energies",
    "orbital_energies_alpha",
    "orbital_energies_beta",
    "gradient",
    "cartesian_hessian",
)
_SCALARS = (bool, int, float, str, type(None))


def _build() -> Any:
    return _plain(getattr(ferric, "__build__", None))


def _molecule_dict(mol: Any) -> dict[str, Any]:
    ghost = [bool(g) for g in mol.is_ghost()]
    return {
        "type": "Molecule",
        "symbols": list(mol.symbols()),
        "coords_bohr": [list(map(float, xyz)) for xyz in mol.coords_bohr()],
        "ghost": ghost,
        "charge": int(mol.charge),
        "multiplicity": int(mol.multiplicity),
        "ferric_build": _build(),
    }


def to_dict(obj: Any, arrays: bool = False) -> dict[str, Any]:
    """A plain dict (JSON-able apart from non-finite floats) for a Molecule or a
    result object. See the module docstring."""
    if isinstance(obj, ferric.Molecule):
        return _molecule_dict(obj)
    out: dict[str, Any] = {"type": type(obj).__name__}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except (AttributeError, NotImplementedError):  # noqa: S112 - not a field
            continue
        if callable(value):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if name == "certificate":
            out[name] = _plain(value)
        elif isinstance(value, _SCALARS):
            out[name] = value
    certificate = getattr(obj, "certificate", None)
    if callable(certificate):
        out["certificate"] = _plain(certificate())
    if "energy" not in out:
        for key in ("total_energy",):
            if isinstance(out.get(key), float):
                out["energy"] = out[key]
                break
    if arrays:
        for name in _ARRAY_ACCESSORS:
            accessor = getattr(obj, name, None)
            if accessor is None:
                continue
            try:
                value = accessor() if callable(accessor) else accessor
            except (RuntimeError, ValueError, TypeError):  # noqa: S112 - not for this result
                continue
            if isinstance(value, (np.ndarray, list, tuple)):
                out[name] = _plain(np.asarray(value, dtype=float))
    out["ferric_build"] = _build()
    return out


def _null_nonfinite(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _null_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_null_nonfinite(v) for v in obj]
    return obj


def to_json(obj: Any, arrays: bool = False, **json_kw: Any) -> str:
    """``json.dumps(to_dict(obj, arrays))`` with NaN and infinities as ``null``.
    Extra keyword arguments (``indent``...) go to ``json.dumps``."""
    return json.dumps(_null_nonfinite(to_dict(obj, arrays)), allow_nan=False, **json_kw)


def molecule_from_dict(
    d: dict[str, Any], charge: int | None = None, multiplicity: int | None = None
) -> ferric.Molecule:
    """A ``ferric.Molecule`` from ``to_dict(molecule)``. Coordinates are taken as
    Bohr, so the round trip is exact. ``charge`` defaults to ``d["charge"]``
    (else 0) and ``multiplicity`` to ``d["multiplicity"]``; a dict from before
    multiplicity was recorded falls back to 1 for an even electron count and 2
    otherwise."""
    if "symbols" not in d or "coords_bohr" not in d:
        raise ValueError("molecule_from_dict needs 'symbols' and 'coords_bohr'")
    if charge is None:
        charge = int(d.get("charge", 0))
    if multiplicity is None:
        multiplicity = d.get("multiplicity")
    kw = {"units": "bohr", "charge": charge, "ghost": d.get("ghost") or None}
    coords = np.asarray(d["coords_bohr"], dtype=float)
    if multiplicity is None:
        multiplicity = default_multiplicity(d["symbols"], charge, kw["ghost"])
    return ferric.Molecule.from_arrays(
        d["symbols"], coords, multiplicity=int(multiplicity), **kw
    )


def read_run_log(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Records of a ferric CLI JSON Lines run log, in file order.

    A malformed line raises ``ValueError`` naming it, except a truncated LAST
    line (a job killed mid-write), which is dropped.
    """
    with open(path, encoding="utf-8") as fh:
        lines = [(i, ln) for i, ln in enumerate(fh, 1) if ln.strip()]
    records = []
    for pos, (num, text) in enumerate(lines):
        try:
            rec = json.loads(text)
        except json.JSONDecodeError as err:
            if pos == len(lines) - 1:
                break
            raise ValueError(f"{path}: line {num} is not valid JSON: {err}") from err
        if not isinstance(rec, dict):
            raise ValueError(f"{path}: line {num} is not a JSON object")  # noqa: TRY004
        records.append(rec)
    return records
