"""`ferric.serialize` reads an explicit schema, with the reflected output.

Roadmap item 13 replaced the `dir()`/`getattr` scan of `to_dict` with
`ferric.serialize.RESULT_SCHEMA`. These cases pin that the change is
output-preserving: the schema of every exported class names exactly the
attributes the reflection read, minus the library's exclusion set
`ferric.serialize.NOT_SERIALIZED` (so a getter added later must be added to
the schema, or to that set on purpose), and `to_dict` / `to_json` of real
results are byte-identical to the reflection rule, kept below as
`_reflected_to_dict` (a verbatim copy of the old `to_dict`). The key order of
each case was recorded from the pre-schema build in
`tests/data/serialize_keys.json`.
"""

from __future__ import annotations

import inspect
import json
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

import ferric.serialize as fs  # noqa: E402
from ferric._util import _plain  # noqa: E402

KEYS = json.loads((Path(__file__).parent / "data" / "serialize_keys.json").read_text())

WATER = "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n"
OH = "2\n\nO 0 0 0\nH 0 0 0.97\n"
H2 = "2\n\nH 0 0 0\nH 0 0 0.74\n"


def _reflected_to_dict(obj: Any, arrays: bool = False) -> dict[str, Any]:
    """The pre-schema `to_dict` (reflection over `dir(obj)`), the oracle."""
    if isinstance(obj, ferric.Molecule):
        return fs._molecule_dict(obj)
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
        elif isinstance(value, fs._SCALARS):
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
        for name in fs._ARRAY_ACCESSORS:
            accessor = getattr(obj, name, None)
            if accessor is None:
                continue
            try:
                value = accessor() if callable(accessor) else accessor
            except (RuntimeError, ValueError, TypeError):  # noqa: S112 - not for this result
                continue
            if isinstance(value, (np.ndarray, list, tuple)):
                out[name] = _plain(np.asarray(value, dtype=float))
    out["ferric_build"] = fs._build()
    return out


def _exported_classes() -> list[type]:
    ext = ferric.ferric
    return [
        cls
        for name in sorted(dir(ext))
        if isinstance(cls := getattr(ext, name), type)
        and not issubclass(cls, BaseException)
        and name != "Molecule"
    ]


def _reflected_fields(cls: type) -> list[str]:
    """Public names of `cls` that are not callable on an instance: getters
    (getset descriptors) and class constants, in `dir` order."""
    names = []
    for name in dir(cls):
        if name.startswith("_"):
            continue
        attr = inspect.getattr_static(cls, name)
        if isinstance(
            attr, (types.GetSetDescriptorType, types.MemberDescriptorType, property)
        ):
            names.append(name)
        elif not callable(getattr(cls, name)):
            names.append(name)
    return names


@pytest.mark.parametrize("cls", _exported_classes(), ids=lambda c: c.__name__)
def test_schema_names_the_reflected_attributes(cls):
    entry = fs.RESULT_SCHEMA.get(cls.__name__)
    assert entry is not None, f"RESULT_SCHEMA has no entry for {cls.__name__}"
    skipped = fs.NOT_SERIALIZED.get(cls.__name__, frozenset())
    assert list(entry.fields) == [f for f in _reflected_fields(cls) if f not in skipped]
    assert list(entry.fields) == sorted(entry.fields)
    assert list(entry.arrays) == [a for a in fs._ARRAY_ACCESSORS if hasattr(cls, a)]


def test_exclusions_name_real_getters():
    classes = {c.__name__: c for c in _exported_classes()}
    for name, skipped in fs.NOT_SERIALIZED.items():
        assert name in classes, name
        assert skipped <= set(_reflected_fields(classes[name])), (name, skipped)


def test_schema_has_no_stale_classes():
    exported = {c.__name__ for c in _exported_classes()}
    assert set(fs.RESULT_SCHEMA) <= exported


def test_non_ferric_objects_still_reflect():
    class Record:
        energy = -1.5
        label = "x"

        def method(self):
            return 1

    assert fs.to_dict(Record()) == {
        "type": "Record",
        "energy": -1.5,
        "label": "x",
        "ferric_build": fs._build(),
    }


@pytest.fixture(scope="module")
def water():
    return ferric.Molecule.from_xyz_string(WATER)


@pytest.fixture(scope="module")
def sto3g():
    return ferric.BasisSet.bundled("sto-3g")


@pytest.fixture(scope="module")
def ri():
    return ferric.BasisSet.bundled("cc-pvdz-ri")


def _cases():
    """Case name -> builder(water, sto3g, ri) of a cheap result."""
    oh = lambda: ferric.Molecule.from_xyz_string(OH, multiplicity=2)  # noqa: E731
    h2 = lambda: ferric.Molecule.from_xyz_string(H2)  # noqa: E731
    return {
        "rhf": lambda w, b, a: ferric.run_rhf(w, b),
        "rhf_symmetry": lambda w, b, a: ferric.run_rhf(w, b, symmetry=True),
        "rhf_gradient": lambda w, b, a: ferric.run_rhf(w, b, with_gradient=True),
        "dft": lambda w, b, a: ferric.run_dft(w, b, functional="pbe"),
        "dft_d3": lambda w, b, a: ferric.run_dft(
            w, b, functional="pbe", dispersion="d3bj"
        ),
        "uhf": lambda w, b, a: ferric.run_uhf(oh(), b),
        "rohf": lambda w, b, a: ferric.run_rohf(oh(), b),
        "uks": lambda w, b, a: ferric.run_uks(oh(), b, functional="pbe"),
        "rimp2": lambda w, b, a: ferric.run_rimp2(w, b, a),
        "mp3": lambda w, b, a: ferric.run_mp3(w, b, a),
        "scs_mp2": lambda w, b, a: ferric.run_scs_mp2(w, b, a),
        "ccsd": lambda w, b, a: ferric.run_ccsd(w, b, a),
        "casci": lambda w, b, a: ferric.run_casci(w, b, 4, (2, 2)),
        "casscf": lambda w, b, a: ferric.run_casscf(w, b, 4, (2, 2)),
        "tddft": lambda w, b, a: ferric.run_tddft(w, b, a, functional="pbe"),
        "optimize": lambda w, b, a: ferric.run_optimize(h2(), "sto-3g"),
        "frequencies": lambda w, b, a: ferric.run_frequencies(h2(), "sto-3g"),
        "basis_set": lambda w, b, a: b,
    }


CASES = _cases()


@pytest.mark.parametrize("case", sorted(CASES))
def test_to_dict_is_the_reflected_output(case, water, sto3g, ri):
    result = CASES[case](water, sto3g, ri)
    for arrays in (False, True):
        new = fs.to_dict(result, arrays=arrays)
        old = _reflected_to_dict(result, arrays=arrays)
        assert json.dumps(new) == json.dumps(old)
        assert fs.to_json(result, arrays=arrays) == json.dumps(
            fs._null_nonfinite(old), allow_nan=False
        )
        assert list(new) == KEYS[f"{case}/arrays={arrays}"]


def test_unknown_ferric_class_is_refused(monkeypatch, sto3g):
    monkeypatch.delitem(fs.RESULT_SCHEMA, "BasisSet")
    with pytest.raises(TypeError, match="no schema for ferric.BasisSet"):
        fs.to_dict(sto3g)
