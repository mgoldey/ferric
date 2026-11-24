"""book.yml runs the derivation proofs a pull request can change, and all of them when unsure.

``scripts/select_proofs.py`` turns the changed files into ``all``, ``none`` or
a list of proof scripts for the book job's proof step. The fail-safes matter
most: a selection that came back ``none`` when it could not tell would skip
every proof with a green job. These tests drive the pure ``select`` function
with made-up cases and registries, so no git history is needed, plus checks
against the real case list. Standard library and pytest only: the book job runs
this file with ``--noconftest`` (conftest.EXTENSION_FREE lists it too).
"""

import copy
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "select_proofs.py"

_spec = importlib.util.spec_from_file_location("select_proofs", SCRIPT)
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)

CASES = ["a_sympy.py", "b_sympy.py", "mp2_variant_identities.py"]
SOURCES = {
    "a_sympy.py": 'SRC = root / "crates" / "ferric-x" / "src" / "kernel.rs"\n',
    "b_sympy.py": "# cites ferric-y/src/other_kernel.rs::f only\n",
    "mp2_variant_identities.py": "",
}
REGISTRY = {
    "_doc": "d",
    "routes": {"cli:a": ["m-a"], "cli:b": ["m-b"]},
    "methods": {
        "m-a": {"book": ["book/src/01-x/01-a.md#a"], "scripts": ["scripts/a_sympy.py"]},
        "m-b": {"book": ["book/src/01-x/02-b.md#b"], "scripts": ["scripts/b_sympy.py"]},
    },
    "gaps": {"g": {"reason": "r", "book": ["book/src/01-x/03-c.md#c"]}},
}


def run(*changed, old=REGISTRY, new=REGISTRY):
    sel = sp.select(list(changed), CASES, SOURCES, old, new)
    return str(sel)


def edited(section=None, mid=None, entry=None):
    """A copy of REGISTRY, with ``REGISTRY[section][mid] = entry`` if given."""
    reg = copy.deepcopy(REGISTRY)
    if section:
        reg[section][mid] = entry
    return reg


def test_prose_only_runs_no_proof():
    assert run("book/src/10-mp2/01-a.md", "book/src/appendix-b-functions.md") == "none"
    assert run("crates/ferric-scf/theory.md", "book/BUILD.md") == "none"


def test_one_script_runs_only_its_proofs():
    assert run("scripts/a_sympy.py") == "a_sympy.py"
    assert run("scripts/mp2_variant_identities.py", "book/src/x.md") == (
        "mp2_variant_identities.py"
    )
    assert run("scripts/b_sympy.py", "scripts/a_sympy.py") == "a_sympy.py b_sympy.py"


def test_runner_checker_selector_and_workflow_run_all():
    for path in (
        "crates/ferric-python/tests/test_derivation_scripts.py",
        "crates/ferric-python/tests/test_derivation_registry.py",
        "crates/ferric-python/tests/test_book_build.py",
        "crates/ferric-python/tests/test_select_proofs.py",
        "scripts/book_build.py",
        "scripts/select_proofs.py",
        ".github/workflows/book.yml",
    ):
        assert run("book/src/x.md", path, "scripts/a_sympy.py") == "all", path


def test_unknown_file_under_scripts_or_derivations_runs_all():
    # A helper module a proof could import, a new script not yet in CASES.
    assert run("scripts/sympy_helpers.py") == "all"
    assert run("scripts/c_sympy.py") == "all"
    assert run("scripts/sub/dir.py") == "all"
    assert run("testdata/derivations/schema.json") == "all"


def test_registry_entry_change_runs_that_entrys_scripts():
    reg = "testdata/derivations/registry.json"
    moved = {"book": ["book/src/01-x/09-z.md#z"], "scripts": ["scripts/b_sympy.py"]}
    new = edited("methods", "m-b", moved)
    assert run(reg, new=new) == "b_sympy.py"
    # A removed method still runs the scripts it named.
    gone = edited()
    del gone["methods"]["m-a"]
    assert run(reg, new=gone) == "a_sympy.py"
    # Routes and the _doc string name no script.
    routes = edited()
    routes["routes"]["cli:c"] = ["m-a"]
    assert run(reg, new=routes) == "none"


def test_registry_shape_change_runs_all():
    reg = "testdata/derivations/registry.json"
    extra_field = edited(
        "methods", "m-a", {**REGISTRY["methods"]["m-a"], "status": "x"}
    )
    assert run(reg, new=extra_field) == "all"
    extra_section = edited()
    extra_section["aliases"] = {}
    assert run(reg, new=extra_section) == "all"
    assert run(reg, old=None) == "all"
    assert run(reg, new=None) == "all"


def test_a_file_a_proof_reads_runs_that_proof():
    assert run("crates/ferric-x/src/kernel.rs") == "a_sympy.py"
    assert run("crates/ferric-y/src/other_kernel.rs") == "b_sympy.py"
    # A whole-token match: "kernel.rs" does not select b's "other_kernel.rs".
    assert run("crates/ferric-z/src/mod.rs") == "none"


def test_a_slow_tier_script_is_never_listed():
    # The job deselects the validation tier; listing it would collect nothing.
    sel = sp.select(
        ["scripts/a_sympy.py", "scripts/b_sympy.py"],
        CASES,
        SOURCES,
        slow=["b_sympy.py"],
    )
    assert str(sel) == "a_sympy.py"
    assert (
        str(sp.select(["scripts/b_sympy.py"], CASES, SOURCES, slow=["b_sympy.py"]))
        == "none"
    )


def test_nothing_listed_runs_all():
    assert run() == "all"


def test_the_real_case_list_is_read():
    cases = sp.read_cases((REPO / sp.RUNNER).read_text(encoding="utf-8"))
    assert len(cases) > 50 and "omega_ri_sympy.py" in cases
    assert "mp2_variant_identities.py" in cases
    slow = sp.read_slow((REPO / sp.RUNNER).read_text(encoding="utf-8"))
    assert 0 < len(slow) < len(cases) and set(slow) <= set(cases)
    assert str(sp.selection_for([f"scripts/{slow[0]}"])) == "none"
    assert str(sp.selection_for(["scripts/omega_ri_sympy.py"])) == "omega_ri_sympy.py"
    assert str(sp.selection_for(["book/src/appendix-b-functions.md"])) == "none"


def test_an_edit_to_a_real_registry_entry_selects_its_scripts():
    # The real registry's entries carry fields beyond the made-up ones above
    # (entry_points); an ordinary edit must still not read as a shape change.
    real = json.loads((REPO / sp.REGISTRY).read_text(encoding="utf-8"))
    cases = sp.read_cases((REPO / sp.RUNNER).read_text(encoding="utf-8"))
    mid, entry = next(
        (k, e)
        for k, e in real["methods"].items()
        if e["scripts"] and all(Path(x).name in cases for x in e["scripts"])
    )
    new = copy.deepcopy(real)
    new["methods"][mid]["book"] = entry["book"] + entry["book"][:1]
    sel = sp.select([sp.REGISTRY], cases, {}, real, new)
    assert sel.mode == "some"
    assert sel.scripts == sorted(Path(x).name for x in entry["scripts"])


def test_every_proof_that_reads_a_file_names_it():
    """The data-dependency rule matches a changed file's base name against the
    proof sources. A proof that opens a file whose name it builds at run time
    would be missed, so each one that reads files must name an existing one."""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split()
    names = {Path(p).name for p in tracked if not p.startswith(("book/", "scripts/"))}
    reads = re.compile(r"read_text|read_bytes|\bopen\(|json\.load|np\.load|loadtxt")
    token = re.compile(r"[\w.-]+\.(?:rs|json|toml|txt|csv|npy|npz|h5|xyz)\b")
    for case in sp.read_cases((REPO / sp.RUNNER).read_text(encoding="utf-8")):
        src = (REPO / "scripts" / case).read_text(encoding="utf-8")
        if reads.search(src):
            assert set(token.findall(src)) & names, (
                f"scripts/{case} reads a file but names no tracked file"
            )


def test_cli_writes_github_output(tmp_path):
    files = tmp_path / "changed.txt"
    files.write_text("book/src/x.md\nscripts/omega_ri_sympy.py\n")
    out = tmp_path / "out"
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--files",
            str(files),
            "--github-output",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert r.stdout.strip() == "omega_ri_sympy.py"
    assert out.read_text() == "mode=some\nscripts=omega_ri_sympy.py\n"


def test_cli_lists_the_slow_tier(tmp_path):
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--slow", "--github-output", str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    slow = sp.read_slow((REPO / sp.RUNNER).read_text(encoding="utf-8"))
    assert slow and json.loads(r.stdout) == slow
    assert out.read_text() == f"slow={json.dumps(slow)}\n"


def test_the_slow_tier_runs_nightly_and_on_dispatch_only():
    # No other workflow runs the runner's validation tier (ci.yml's
    # validation pytest ignores the runner), so book.yml's slow-proofs jobs
    # are its only coverage.
    wf = (REPO / ".github/workflows/book.yml").read_text(encoding="utf-8")
    job = wf[wf.index("\n  slow-proofs-list:") : wf.index("\n  pdf:")]
    assert (
        "if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'"
        in job
    )
    assert "select_proofs.py --slow" in job
    assert "fromJSON(needs.slow-proofs-list.outputs.slow)" in job
    assert "-m validation" in job and "FERRIC_PROOF_SCRIPTS:" in job
