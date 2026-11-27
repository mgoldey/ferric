"""testdata/derivations/registry.json covers every method under the derivations rule.

CONTRIBUTING.md, "Every method has a derivation and a sympy proof": each method
has a derivation in a committed book section (``book/src``) and a sympy proof
in ``scripts/*_sympy.py`` that a test under this directory runs.

The method list is DERIVED from the code, never hand-listed:

* ``cli:<kind>`` for every entry of ``SUPPORTED_METHOD_KINDS``
  (``crates/ferric-cli/src/lib.rs``);
* ``optimize:<kind>`` for every arm of the CLI ``run_optimize`` dispatch (the
  analytic-gradient routes) and ``frequencies:<kind>`` for every arm mapping to
  a ``FrequencyReference`` in ``run_frequencies`` (the analytic-Hessian routes);
* ``md:<kind>`` for every entry of ``MD_KINDS``
  (``crates/ferric-cli/src/config.rs``), the kinds ``task = "md"`` runs;
* ``py:<name>`` for every top-level ``def run_*`` or ``def tune_*`` in
  ``crates/ferric-python/python/ferric/ferric.pyi``, plus every top-level response,
  derivative or coupling function (name containing ``polariz``,
  ``gradient``, ``hessian`` or ``coupling``) or ending in ``_energy``;
* ``py-gradient:<name>`` and ``py-stress:<name>`` for every periodic entry
  point (``*_gamma``, ``*_kpts``) that takes ``with_gradient`` or
  ``with_stress`` (the molecular gradients are their ``optimize:`` routes);
* ``cell:<route>`` for every ``PeriodicRoute`` of the CLI ``[cell]`` section
  (``crates/ferric-cli/src/config.rs``), and ``cell-optimize:<route>`` and
  ``cell-stress:<route>`` for every route with a ``k_force_*`` or
  ``k_stress_*`` arm in ``crates/ferric-cli/src/periodic.rs``.

Every derived route must appear under ``routes``, mapping to one or more method
ids. Each method id is either in ``methods`` (both pieces present: the book
section exists and each script exists and is run by a test here) or in ``gaps``
(with a reason). A gap may record the pieces it already has; the test fails if
a gap holds both, so the gap list only shrinks.

Mutation: delete one ``routes`` entry (e.g. ``"cli:rhf"`` or ``"md:rhf"``)
from the manifest; ``test_every_derived_route_is_in_the_manifest`` fails.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "testdata" / "derivations" / "registry.json"
CLI_LIB = ROOT / "crates" / "ferric-cli" / "src" / "lib.rs"
CLI_CONFIG = ROOT / "crates" / "ferric-cli" / "src" / "config.rs"
CLI_PERIODIC = ROOT / "crates" / "ferric-cli" / "src" / "periodic.rs"
PYI = ROOT / "crates" / "ferric-python" / "python" / "ferric" / "ferric.pyi"
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
    md = re.search(r"MD_KINDS: &\[&str\] = &\[(.*?)\];", CLI_CONFIG.read_text(), re.S)
    routes |= {f"md:{k}" for k in re.findall(r'"([^"]+)"', md.group(1))}
    pyi = PYI.read_text()
    sig = re.compile(r"^def ([A-Za-z_][A-Za-z0-9_]*)\((.*?)\)\s*->", re.M | re.S)
    for name, args in sig.findall(pyi):
        if (
            name.startswith(("run_", "tune_"))
            or re.search("polariz|gradient|hessian|coupling", name)
            or name.endswith("_energy")
        ):
            routes.add(f"py:{name}")
            periodic = re.search(r"_(gamma|kpts)$", name)
            if periodic and re.search(r"\bwith_gradient\b", args):
                routes.add(f"py-gradient:{name}")
            if periodic and re.search(r"\bwith_stress\b", args):
                routes.add(f"py-stress:{name}")
    routes |= _cell_routes()
    return routes


def _cell_routes():
    """The periodic CLI routes: one per ``PeriodicRoute`` variant, plus the
    variants the k-point force and stress dispatch accept."""
    enum = re.search(r"pub enum PeriodicRoute \{(.*?)\}", CLI_CONFIG.read_text(), re.S)
    routes = {f"cell:{v.lower()}" for v in re.findall(r"(\w+),", enum.group(1))}
    src = CLI_PERIODIC.read_text()
    for prefix, kind in (("k_force_", "cell-optimize"), ("k_stress_", "cell-stress")):
        arm = re.compile(rf"((?:PeriodicRoute::\w+\s*\|?\s*)+)=>\s*{prefix}")
        for m in arm.finditer(src):
            for v in re.findall(r"PeriodicRoute::(\w+)", m.group(1)):
                routes.add(f"{kind}:{v.lower()}")
    return routes


def _registry():
    return json.loads(REGISTRY.read_text())


def _runner_sources():
    return {p.name: p.read_text() for p in TESTS.glob("test_*.py")}


def _book_ok(ref):
    """``book/src/<page>.md#<label>``: the page exists and has a heading ``{#label}``."""
    path, _, label = ref.partition("#")
    md = ROOT / path
    return (
        path.startswith("book/src/")
        and md.is_file()
        and bool(label)
        and re.search(
            rf"^#{{1,6}} .*\{{#{re.escape(label)}\}}\s*$", md.read_text(), re.M
        )
        is not None
    )


def _script_ok(script, runners):
    name = Path(script).name
    return (ROOT / script).is_file() and any(name in s for s in runners.values())


def test_derivation_is_not_vacuous():
    routes = derived_routes()
    # The parse must see the CLI kinds, the analytic gradient and Hessian
    # arms and the Python entry points; a broken regex returns an empty set.
    for must in (
        "cli:rhf",
        "optimize:rimp2",
        "frequencies:uhf",
        "md:ksdft",
        "py:run_rhf",
        "py:d3bj_energy",
        "py:tune_omega",
        "py:cdft_coupling",
        "py-gradient:run_rhf_gamma",
        "py-stress:run_uks_kpts",
        "cell:drpa",
        "cell-optimize:rohf",
        "cell-stress:roks",
    ):
        assert must in routes, must
    # The periodic correlation routes have no analytic force or stress.
    assert "cell-optimize:mp2" not in routes
    assert "py-stress:run_mp2_kpts" not in routes
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
        assert entry["book"], mid
        for ref in entry["book"]:
            assert _book_ok(ref), f"{mid}: book section {ref} does not exist"
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
        book = [r for r in gap.get("book", []) if _book_ok(r)]
        scripts = [s for s in gap.get("scripts", []) if _script_ok(s, runners)]
        assert len(book) == len(gap.get("book", [])), f"{mid}: stale book ref"
        assert not (book and scripts), (
            f"gap {mid} has a book section and a run script: move it to methods"
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


def test_book_ref_check_rejects_a_dangling_label():
    """Mutation: a label that is not a heading of the page, or a non-book path."""
    good = next(iter(_registry()["methods"].values()))["book"][0]
    assert _book_ok(good)
    page = good.partition("#")[0]
    assert not _book_ok(page + "#no-such-label")
    assert not _book_ok(
        "crates/ferric-scf/theory.md#hartree-fock-energy-functionals-rhf-uhf-rohf"
    )
