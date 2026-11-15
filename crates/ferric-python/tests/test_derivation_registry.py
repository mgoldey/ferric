"""testdata/derivations/registry.json covers every method under the derivations rule.

CONTRIBUTING.md, "Every method has a derivation and a sympy proof": each method
ships a LaTeX derivation in its crate's ``theory.md`` and a sympy proof in
``scripts/*_sympy.py`` that a test under this directory runs.

The method list is DERIVED from the code, never hand-listed:

* ``cli:<kind>`` for every entry of ``SUPPORTED_METHOD_KINDS``
  (``crates/ferric-cli/src/lib.rs``);
* ``optimize:<kind>`` for every arm of the CLI ``run_optimize`` dispatch (the
  analytic-gradient routes) and ``frequencies:<kind>`` for every arm mapping to
  a ``FrequencyReference`` in ``run_frequencies`` (the analytic-Hessian routes);
* ``py:<name>`` for every top-level ``def run_*`` in
  ``crates/ferric-python/ferric.pyi``, plus every top-level response or
  derivative function (name containing ``polariz``, ``gradient`` or
  ``hessian``).

Every derived route must appear under ``routes``, mapping to one or more method
ids. Each method id is either in ``methods`` (both pieces present: the theory
anchor exists and each script exists and is run by a test here) or in ``gaps``
(with a reason). A gap may record the pieces it already has; the test fails if
a gap holds both, so the gap list only shrinks.

Mutation: delete one ``routes`` entry (e.g. ``"cli:rhf"``) from the manifest;
``test_every_derived_route_is_in_the_manifest`` fails.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "testdata" / "derivations" / "registry.json"
CLI_LIB = ROOT / "crates" / "ferric-cli" / "src" / "lib.rs"
PYI = ROOT / "crates" / "ferric-python" / "ferric.pyi"
TESTS = Path(__file__).resolve().parent


def _fn_body(src, name):
    start = src.index(f"\nfn {name}(")
    end = re.compile(r"\n(?:pub )?fn |\n///").search(src, start + 1)
    return src[start : end.start() if end else len(src)]


def derived_routes():
    """The method list, read from the CLI source and the Python stub."""
    src = CLI_LIB.read_text()
    block = re.search(r"SUPPORTED_METHOD_KINDS: &\[&str\] = &\[(.*?)\];", src, re.S)
    routes = {f"cli:{k}" for k in re.findall(r'"([^"]+)"', block.group(1))}
    arm = re.compile(r'^ {8}("[^"]+"(?:\s*\|\s*"[^"]+")*)\s*=>(.*)$', re.M)
    for m in arm.finditer(_fn_body(src, "run_optimize")):
        routes |= {f"optimize:{k}" for k in re.findall(r'"([^"]+)"', m.group(1))}
    for m in arm.finditer(_fn_body(src, "run_frequencies")):
        if "FrequencyReference::" in m.group(2):
            routes |= {f"frequencies:{k}" for k in re.findall(r'"([^"]+)"', m.group(1))}
    pyi = PYI.read_text()
    for name in re.findall(r"^def ([A-Za-z_][A-Za-z0-9_]*)\(", pyi, re.M):
        if name.startswith("run_") or re.search("polariz|gradient|hessian", name):
            routes.add(f"py:{name}")
    return routes


def _registry():
    return json.loads(REGISTRY.read_text())


def _slug(heading):
    s = heading.strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def _anchors(md_path):
    text = md_path.read_text()
    anchors = set()
    counts = {}
    for h in re.findall(r"^#{1,6} (.+)$", text, re.M):
        base = _slug(h)
        n = counts.get(base, 0)
        anchors.add(base if n == 0 else f"{base}-{n}")
        counts[base] = n + 1
    anchors |= set(re.findall(r'<a id="([^"]+)"', text))
    return anchors


def _runner_sources():
    return {p.name: p.read_text() for p in TESTS.glob("test_*.py")}


def _theory_ok(ref):
    path, _, anchor = ref.partition("#")
    md = ROOT / path
    return md.is_file() and anchor in _anchors(md)


def _script_ok(script, runners):
    name = Path(script).name
    return (ROOT / script).is_file() and any(name in s for s in runners.values())


def test_derivation_is_not_vacuous():
    routes = derived_routes()
    # The parse must see the CLI kinds, the analytic gradient and Hessian
    # arms and the Python entry points; a broken regex returns an empty set.
    for must in ("cli:rhf", "optimize:rimp2", "frequencies:uhf", "py:run_rhf"):
        assert must in routes, must
    assert len(routes) > 90


def test_every_derived_route_is_in_the_manifest():
    missing = sorted(derived_routes() - set(_registry()["routes"]))
    assert not missing, f"routes missing from {REGISTRY.name}: {missing}"


def test_manifest_has_no_stale_routes():
    stale = sorted(set(_registry()["routes"]) - derived_routes())
    assert not stale, f"routes no longer in the code: {stale}"


def test_every_method_id_is_either_covered_or_a_gap():
    reg = _registry()
    methods, gaps = set(reg["methods"]), set(reg["gaps"])
    assert not methods & gaps, sorted(methods & gaps)
    used = {m for ms in reg["routes"].values() for m in ms}
    assert used <= methods | gaps, sorted(used - methods - gaps)
    for route, ms in reg["routes"].items():
        assert ms, f"{route} maps to no method"


def test_covered_methods_have_both_pieces():
    runners = _runner_sources()
    for mid, entry in _registry()["methods"].items():
        assert entry["theory"], mid
        for ref in entry["theory"]:
            assert _theory_ok(ref), f"{mid}: theory anchor {ref} does not exist"
        assert entry["scripts"], mid
        for script in entry["scripts"]:
            assert _script_ok(script, runners), (
                f"{mid}: {script} is missing or no test under "
                "crates/ferric-python/tests runs it"
            )


def test_gaps_only_shrink():
    runners = _runner_sources()
    for mid, gap in _registry()["gaps"].items():
        assert gap["reason"].strip(), mid
        theory = [r for r in gap.get("theory", []) if _theory_ok(r)]
        scripts = [s for s in gap.get("scripts", []) if _script_ok(s, runners)]
        assert len(theory) == len(gap.get("theory", [])), f"{mid}: stale theory ref"
        assert not (theory and scripts), (
            f"gap {mid} has a theory section and a run script: move it to methods"
        )


def test_runners_are_collected_without_the_extension():
    """conftest.py ignores every test module when ferric is missing, except
    the EXTENSION_FREE ones; a proof runner outside that set would pass by
    running nothing."""
    conftest = (TESTS / "conftest.py").read_text()
    block = re.search(r"EXTENSION_FREE = frozenset\((.*?)\)\n", conftest, re.S)
    exempt = set(re.findall(r'"([^"]+\.py)"', block.group(1)))
    runners = _runner_sources()
    scripts = {
        Path(s).name
        for entry in _registry()["methods"].values()
        for s in entry["scripts"]
    }
    for script in scripts:
        hosts = {n for n, src in runners.items() if script in src} - {
            "test_derivation_registry.py"
        }
        assert hosts & exempt, f"{script} is run only by {sorted(hosts)}"
