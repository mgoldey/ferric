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


def generate(module=None) -> dict:
    if module is None:
        import ferric as module
    functions, classes, constants = {}, {}, {}
    for name in sorted(n for n in dir(module) if not n.startswith("_")):
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
    return {"functions": functions, "classes": classes, "constants": constants}


def render(snapshot: dict) -> str:
    return json.dumps(snapshot, indent=1, sort_keys=True) + "\n"


def main(argv: list[str]) -> int:
    text = render(generate())
    if "--check" in argv:
        return 0 if SNAPSHOT.read_text() == text else 1
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(text)
    print(f"wrote {SNAPSHOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
