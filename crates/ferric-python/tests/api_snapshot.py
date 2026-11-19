"""Introspect the public `ferric` API into a deterministic, sorted dict.

Used by `test_api_snapshot.py` (compare) and, as a script, to regenerate the
committed snapshot:

    python crates/ferric-python/tests/api_snapshot.py            # rewrite
    python crates/ferric-python/tests/api_snapshot.py --check    # exit 1 on drift

Captured: for every public (no leading underscore) callable in the module its
parameters (name, kind, default repr); for every public class its sorted public
attribute names, plus a signature for the constructor and each callable member
that `inspect` can read (tagged instance / static / class method); and the
public numeric constants. Submodules are skipped, and so is anything `inspect`
cannot handle, deterministically (recorded as `null`). Build info and version
strings are private (`__version__`, `build_info()` values) and never captured.

The pure-Python submodules (`ferric.serialize`, `.rdkit`, `.qcschema`, `.units`,
`.molden`, `.fcidump`, `.viz`, `.ase`) go under the top-level key `submodules`:
each is captured by its `__all__` with the same function / class / constant
layout (a function's parameters, a constant's repr). `ferric.ase` imports ASE
at module level, so it is recorded only when importable, and
`test_api_snapshot.py` skips its section otherwise (`optional_missing()`);
there only `FerricCalculator.implemented_properties` and `.default_parameters`
are captured (the rest of the class is ASE's own, version-dependent).
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

SNAPSHOT = Path(__file__).resolve().parent / "data" / "api_snapshot.json"


def _sig(obj) -> list[dict] | None:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    out = []
    for p in sig.parameters.values():
        d = {"name": p.name, "kind": p.kind.name}
        if p.default is not inspect.Parameter.empty:
            d["default"] = repr(p.default)
        out.append(d)
    return out


def _member_kind(cls, name: str) -> str:
    raw = inspect.getattr_static(cls, name)
    if isinstance(raw, staticmethod):
        return "static"
    if isinstance(raw, classmethod):
        return "class"
    if callable(getattr(cls, name, None)) and not isinstance(raw, (property, type)):
        # pyo3 #[staticmethod]/#[classmethod] surface as builtin methods bound
        # to the class or None; instance methods are method descriptors.
        bound = getattr(cls, name)
        owner = getattr(bound, "__self__", None)
        if owner is cls:
            return "class"
        if inspect.isbuiltin(bound) and owner is None:
            return "static"
        return "method"
    return "attribute"


SUBMODULES = (
    "serialize",
    "rdkit",
    "qcschema",
    "units",
    "molden",
    "fcidump",
    "viz",
    "ase",
)


def _calculator_entry(module) -> dict:
    cls = module.FerricCalculator
    return {
        "functions": {},
        "classes": {
            "FerricCalculator": {
                "implemented_properties": list(cls.implemented_properties),
                "default_parameters": {
                    k: repr(v) for k, v in sorted(cls.default_parameters.items())
                },
            }
        },
        "constants": {},
    }


def _import_submodule(name: str):
    """The `ferric.<name>` module, or `None` when an optional dependency of
    its module-level imports (only `ase`) is missing."""
    import importlib
    import importlib.util

    if name == "ase" and importlib.util.find_spec("ase") is None:
        return None
    try:
        return importlib.import_module(f"ferric.{name}")
    except ModuleNotFoundError as exc:
        # Only the optional dependency itself being absent is tolerated; a
        # broken internal import in ferric/ase.py must raise.
        if name == "ase" and exc.name is not None and exc.name.split(".")[0] == "ase":
            return None
        raise


def optional_missing() -> list[str]:
    """Submodules whose section cannot be generated here (deterministic)."""
    return [n for n in SUBMODULES if _import_submodule(n) is None]


def drop_optional(data: dict) -> list[str]:
    """Drop from a committed snapshot the sections that cannot be generated
    here (optional dependency missing); returns the dropped names."""
    skipped = optional_missing()
    for name in skipped:
        data.get("submodules", {}).pop(name, None)
    return skipped


def generate_submodule(name: str) -> dict | None:
    module = _import_submodule(name)
    if module is None:
        return None
    if name == "ase":
        return _calculator_entry(module)
    return generate(module, names=module.__all__)


def generate(module=None, names=None) -> dict:
    top = module is None
    if top:
        import ferric as module
    functions, classes, constants = {}, {}, {}
    if names is None:
        names = [n for n in dir(module) if not n.startswith("_")]
    for name in sorted(names):
        obj = getattr(module, name)
        if inspect.ismodule(obj):
            continue
        if inspect.isclass(obj):
            members = {}
            for m in sorted(n for n in dir(obj) if not n.startswith("_")):
                kind = _member_kind(obj, m)
                entry = {"kind": kind}
                if kind != "attribute":
                    entry["signature"] = _sig(getattr(obj, m))
                members[m] = entry
            classes[name] = {"constructor": _sig(obj), "members": members}
        elif callable(obj):
            functions[name] = _sig(obj)
        elif isinstance(obj, (int, float, str, bool)):
            constants[name] = repr(obj)
    out = {"functions": functions, "classes": classes, "constants": constants}
    if top:
        subs = {n: generate_submodule(n) for n in SUBMODULES}
        out["submodules"] = {n: v for n, v in subs.items() if v is not None}
    return out


def render(snapshot: dict) -> str:
    return json.dumps(snapshot, indent=1, sort_keys=True) + "\n"


def main(argv: list[str]) -> int:
    missing = optional_missing()
    if missing and "--check" not in argv:
        print(f"cannot regenerate without {missing}: install the optional deps")
        return 2
    text = render(generate())
    if "--check" in argv:
        if SNAPSHOT.read_text() == text:
            return 0
        committed = json.loads(SNAPSHOT.read_text())
        drop_optional(committed)
        return 0 if committed == json.loads(text) else 1
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(text)
    print(f"wrote {SNAPSHOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
