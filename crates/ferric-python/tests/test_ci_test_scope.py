"""CI runs the TIERED tests of the crates a change could break, and all of them when unsure.

`scripts/ci_test_scope.py` turns the changed files into a nextest filterset for
test-shard's "Tiered tests" step. The fail-safes matter most: a scope that came
back `none()` when it could not tell would drop every tiered test with a green
run. These tests drive the pure `classify` / `readers` functions with made-up
packages and literals, so nothing is compiled and no git repo is needed, plus
one end-to-end call for the unknown-base fallback.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "ci_test_scope.py"

sys.path.insert(0, str(SCRIPT.parent))
_spec = importlib.util.spec_from_file_location("ci_test_scope", SCRIPT)
scope = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scope)

PACKAGES = {"crates/core": "lib-core", "crates/cli": "app-cli", "crates/pbc": "pbc"}
LITERALS = {
    "app-cli": {"examples", "site/src/", "README.md"},
    "pbc": {"testdata/reference/{mol}_{basis}.json", "prototypes/pbc/notes.txt"},
    "lib-core": {"testdata/molecules/water.xyz", "a plain message"},
}


def run(*changes):
    return scope.classify(list(changes), PACKAGES, LITERALS)


def test_a_package_file_selects_its_rdeps():
    # The package NAME, not its directory, goes into rdeps().
    assert run(("M", "crates/cli/src/main.rs"))[0] == "rdeps(app-cli)"


def test_several_packages_are_or_ed_in_name_order():
    expr, _ = run(("M", "crates/pbc/src/lib.rs"), ("M", "crates/core/src/x.rs"))
    assert expr == "rdeps(lib-core) | rdeps(pbc)"


def test_compile_config_and_root_manifest_select_everything():
    for path in (
        "Cargo.lock",
        "Cargo.toml",
        ".cargo/config.toml",
        "rust-toolchain.toml",
    ):
        assert run(("M", path))[0] == "all()", path


def test_a_deleted_package_manifest_selects_everything():
    assert run(("D", "crates/pbc/Cargo.toml"))[0] == "all()"


def test_an_outside_file_selects_the_packages_that_read_it():
    # Named by file name.
    assert run(("M", "README.md"))[0] == "rdeps(app-cli)"
    # Its ancestor directory is a literal (a directory walk / .join chain).
    assert run(("M", "examples/water.toml"))[0] == "rdeps(app-cli)"
    assert run(("M", "site/src/methods/scf.md"))[0] == "rdeps(app-cli)"
    # A name formatted inside an ancestor directory.
    assert run(("M", "testdata/reference/h2o_sto3g.json"))[0] == "rdeps(pbc)"
    # An exact file literal.
    assert run(("M", "testdata/molecules/water.xyz"))[0] == "rdeps(lib-core)"


def test_a_sibling_file_in_a_read_directory_is_not_selected_by_its_neighbour():
    # pbc names prototypes/pbc/notes.txt, not the directory: another file there
    # is not read by it.
    assert run(("M", "prototypes/pbc/other.py"))[0] == "none()"


def test_an_unread_outside_file_selects_nothing():
    assert run(("M", "wiki/notes.md"))[0] == "none()"
    assert run(("M", "scripts/complexity_baseline.json"))[0] == "none()"


def test_no_changes_select_nothing():
    assert run()[0] == "none()"


def test_an_unknown_base_selects_everything():
    for base in ("", "0" * 40, "f" * 40):
        out = subprocess.run(
            [sys.executable, str(SCRIPT), "--base", base],
            capture_output=True,
            text=True,
            cwd=REPO,
            check=True,
        )
        assert out.stdout.strip() == "all()", (base, out.stderr)
