"""Every function and class the module registers must be in `ferric.pyi`,
with the same parameter names in the same order.

`test_pyi_matches_bindings.py` checks a hand-picked REQUIRED list, so a binding
nobody added to that list could be missing from the stub forever. Four were:
`run_tddft` (and its `TddftResult`), `run_double_hybrid`, `run_lmp2_direct`,
and four signatures (`run_uhf`, `run_rohf`, `run_mp3`, `run_laplace_mp2`)
lacked the `memory_budget_gb` kwarg the binding accepts.

This file reads the SOURCE (`src/lib.rs`) rather than the compiled module, so
it runs without a build and sees exactly what the next build will export:

* the functions are the `wrap_pyfunction!(name, m)` registrations;
* the classes are the `add_class::<RustName>()` registrations, mapped to their
  Python names via `#[pyo3(name = "...")]`;
* a function's parameters come from its `#[pyo3(signature = (...))]`, or, if
  it has none, from the Rust parameter list minus `py: Python`.

Private entry points (leading underscore, e.g. `_cli_main`) are exempt.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CRATE = Path(__file__).resolve().parents[1]
SRC = CRATE / "src" / "lib.rs"
PYI = CRATE / "ferric.pyi"


def _split_top(body: str) -> list[str]:
    """Split on commas not nested inside (), [], {} or <>."""
    out, depth, cur = [], 0, ""
    for ch in body:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [o.strip() for o in out if o.strip()]


def _balanced(text: str, open_idx: int) -> str:
    """Contents of the bracket group whose opener is at text[open_idx]."""
    depth, j = 1, open_idx + 1
    while depth:
        if text[j] in "([":
            depth += 1
        elif text[j] in ")]":
            depth -= 1
        j += 1
    return text[open_idx + 1 : j - 1]


@pytest.fixture(scope="module")
def src() -> str:
    return SRC.read_text()


@pytest.fixture(scope="module")
def stub() -> str:
    return PYI.read_text()


def _exported_functions(src: str) -> list[str]:
    names = re.findall(r"wrap_pyfunction!\(\s*(\w+)", src)
    return [n for n in names if not n.startswith("_")]


def _rust_params(src: str, name: str) -> list[str]:
    for chunk in src.split("#[pyfunction]")[1:]:
        m = re.search(r"\bfn\s+(\w+)", chunk)
        if not m or m.group(1) != name:
            continue
        head = chunk[: m.start()]
        sig = re.search(r"signature\s*=\s*\(", head)
        if sig:
            parts = _split_top(_balanced(head, sig.end() - 1))
            return [re.split(r"[=\s]", p)[0] for p in parts if p not in ("*", "/")]
        params = _split_top(_balanced(chunk, chunk.index("(", m.end())))
        names = []
        for p in params:
            p = re.sub(r"#\[[^\]]*\]\s*", "", p)
            n, _, t = p.partition(":")
            if "Python<" in t:
                continue
            names.append(n.strip().replace("mut ", ""))
        return names
    raise AssertionError(f"no #[pyfunction] fn {name} in {SRC}")


def _stub_params(stub: str, name: str) -> list[str] | None:
    m = re.search(rf"^def {re.escape(name)}\(", stub, re.M)
    if not m:
        return None
    body = "\n".join(line.split("#")[0] for line in _balanced(stub, m.end() - 1).splitlines())
    return [p.split(":")[0].split("=")[0].strip() for p in _split_top(body) if p not in ("*", "/")]


def _exported_classes(src: str) -> list[str]:
    py_name = {}
    for m in re.finditer(r"((?:#\[[^\]]*\]\s*)+)(?:pub\s+)?struct\s+(\w+)", src):
        attrs, rust = m.groups()
        if "pyclass" in attrs:
            n = re.search(r'name\s*=\s*"(\w+)"', attrs)
            py_name[rust] = n.group(1) if n else rust
    return [py_name.get(c, c) for c in re.findall(r"add_class::<\s*(\w+)\s*>", src)]


def test_the_scan_finds_the_module(src):
    """Reachability: a regex that matches nothing would pass every check below."""
    assert len(_exported_functions(src)) > 40
    assert len(_exported_classes(src)) > 25
    assert "run_rhf" in _exported_functions(src)
    assert "RhfResult" in _exported_classes(src)


def test_every_registered_function_is_in_the_stub(src, stub):
    missing = [n for n in _exported_functions(src) if not re.search(rf"^def {n}\(", stub, re.M)]
    assert not missing, f"registered in lib.rs but absent from ferric.pyi: {missing}"


def test_every_registered_class_is_in_the_stub(src, stub):
    missing = [c for c in _exported_classes(src) if not re.search(rf"^class {c}\b", stub, re.M)]
    assert not missing, f"registered in lib.rs but absent from ferric.pyi: {missing}"


def test_stub_parameters_match_the_binding(src, stub):
    diffs = []
    for name in _exported_functions(src):
        s = _stub_params(stub, name)
        if s is None:
            continue  # reported by test_every_registered_function_is_in_the_stub
        r = _rust_params(src, name)
        if r != s:
            diffs.append(f"{name}:\n    binding {r}\n    stub    {s}")
    assert not diffs, "ferric.pyi parameter lists differ from the bindings:\n" + "\n".join(diffs)
