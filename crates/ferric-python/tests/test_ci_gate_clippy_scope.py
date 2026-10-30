"""The push gate lints the changed crates, and anything uncertain lints them all.

`scripts/ci_gate_clippy_scope.py` picks the clippy scope for `scripts/ci-gate.sh`.
The tests that matter most are the fail-safes: a selector that returned "nothing
to lint" when it could not tell would lose clippy silently, with a green gate.

Everything runs against a throwaway workspace with a local bare `origin`, whose
package names differ from their directories on purpose. The gate tests run the
real ci-gate.sh with a stub `cargo` that records its clippy arguments, a private
lock file, and every other step skipped. No crate is compiled.
"""

import importlib.util
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "scripts"
SELECTOR = SCRIPTS / "ci_gate_clippy_scope.py"
REAL_CARGO = shutil.which("cargo")

pytestmark = pytest.mark.skipif(not REAL_CARGO, reason="needs cargo")


def _env():
    return {
        k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "CI_GATE_"))
    }


def _git(cwd, *args):
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
        env=_env(),
    ).stdout.strip()


def _write(root, rel, text="x\n"):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _add_crate(root, directory, name):
    _write(
        root,
        f"{directory}/Cargo.toml",
        f'[package]\nname = "{name}"\nversion = "0.1.0"\n',
    )
    _write(root, f"{directory}/src/lib.rs", "")


def _commit(root, message="change"):
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)


@pytest.fixture
def repo(tmp_path):
    """A clone on a feature branch, level with origin/main."""
    origin, work = tmp_path / "origin.git", tmp_path / "work"
    subprocess.run(
        ["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True
    )
    work.mkdir()
    _git(work, "init", "-q", "-b", "main")
    _git(work, "config", "user.email", "t@example.com")
    _git(work, "config", "user.name", "t")
    _git(work, "remote", "add", "origin", str(origin))
    _write(work, "Cargo.toml", '[workspace]\nmembers = ["crates/*"]\nresolver = "2"\n')
    _write(work, "Cargo.lock", "# lock\n")
    _write(work, ".gitignore", "target/\n")
    _write(work, "README.md")
    _write(work, "testdata/a.xyz")
    _write(work, "scripts/tool.sh")
    _add_crate(work, "crates/alpha", "alpha-pkg")
    _add_crate(work, "crates/beta", "beta-pkg")
    _add_crate(work, "crates/gamma", "gamma-pkg")
    _commit(work, "init")
    _git(work, "push", "-q", "origin", "main")
    _git(work, "checkout", "-q", "-b", "feature")
    return work


def _select(repo, *args):
    r = subprocess.run(
        ["python3", str(SELECTOR), *args],
        cwd=repo,
        env=_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    mode, crates, reason = r.stdout.split("\n")[:3]
    return mode, crates.split(), reason


# ---- selection ------------------------------------------------------------


def test_one_crate_uses_the_package_name_not_the_directory(repo):
    _write(repo, "crates/alpha/src/lib.rs", "// edit\n")
    _commit(repo)
    assert _select(repo)[:2] == ("crates", ["alpha-pkg"])


def test_two_crates(repo):
    _write(repo, "crates/alpha/src/lib.rs", "// edit\n")
    _write(repo, "crates/beta/src/lib.rs", "// edit\n")
    _commit(repo)
    assert _select(repo)[:2] == ("crates", ["alpha-pkg", "beta-pkg"])


def test_a_file_in_a_crates_tests_dir_selects_that_crate(repo):
    _write(repo, "crates/beta/tests/new_test.rs", "#[test] fn t() {}\n")
    _commit(repo)
    assert _select(repo)[:2] == ("crates", ["beta-pkg"])


def test_docs_and_scripts_only_lint_nothing_and_say_why(repo):
    _write(repo, "README.md", "edit\n")
    _write(repo, "scripts/tool.sh", "edit\n")
    _write(repo, "site/src/page.md", "edit\n")
    _commit(repo)
    mode, crates, reason = _select(repo)
    assert (mode, crates) == ("none", [])
    assert "no compile inputs" in reason


def test_a_branch_level_with_the_base_changed_nothing(repo):
    assert _select(repo) == ("none", [], "nothing changed against the base")


@pytest.mark.parametrize(
    "path",
    [
        "Cargo.toml",
        "Cargo.lock",
        ".cargo/config.toml",
        "clippy.toml",
        "pyproject.toml",
        "some-new-top-level-file",
        "crates/alpha/clippy.toml",
    ],
)
def test_a_compile_input_or_unknown_path_lints_the_workspace(repo, path):
    old = (repo / path).read_text() if (repo / path).exists() else ""
    _write(repo, path, old + "# edit\n")
    _write(repo, "crates/alpha/src/lib.rs", "// edit\n")
    _commit(repo)
    mode, crates, reason = _select(repo)
    assert (mode, crates) == ("workspace", [])
    assert path in reason


def test_no_merge_base_lints_the_workspace(repo):
    # origin/main becomes an unrelated history.
    other = repo.parent / "other"
    other.mkdir()
    _git(other, "init", "-q", "-b", "main")
    _git(other, "config", "user.email", "t@example.com")
    _git(other, "config", "user.name", "t")
    _write(other, "unrelated.txt")
    _commit(other)
    _git(other, "push", "-q", "--force", str(repo.parent / "origin.git"), "main")
    mode, crates, reason = _select(repo)
    assert (mode, crates) == ("workspace", [])
    assert "merge-base" in reason


def test_a_failed_fetch_lints_the_workspace(repo):
    _write(repo, "README.md", "edit\n")
    _commit(repo)
    _git(repo, "remote", "set-url", "origin", str(repo.parent / "gone.git"))
    mode, crates, reason = _select(repo)
    assert (mode, crates) == ("workspace", [])
    assert "fetch" in reason


def test_a_failed_cargo_metadata_lints_the_workspace(repo):
    _write(repo, "crates/alpha/Cargo.toml", "this is not toml [[[\n")
    _commit(repo)
    mode, crates, reason = _select(repo)
    assert (mode, crates) == ("workspace", [])
    assert "metadata" in reason


def test_a_deleted_crate_is_not_passed_to_cargo(repo):
    _write(repo, "crates/alpha/src/lib.rs", "// edit\n")
    shutil.rmtree(repo / "crates/gamma")
    _commit(repo)
    assert _select(repo)[:2] == ("crates", ["alpha-pkg"])


def test_a_renamed_crate_selects_only_the_new_name(repo):
    _git(repo, "mv", "crates/gamma", "crates/delta")
    text = (repo / "crates/delta/Cargo.toml").read_text()
    _write(repo, "crates/delta/Cargo.toml", text.replace("gamma-pkg", "delta-pkg"))
    _commit(repo)
    assert _select(repo)[:2] == ("crates", ["delta-pkg"])


def test_a_real_crate_removal_changes_the_lock_so_lints_the_workspace(repo):
    shutil.rmtree(repo / "crates/gamma")
    _write(repo, "Cargo.lock", "# lock without gamma\n")
    _commit(repo)
    assert _select(repo)[0] == "workspace"


def test_a_file_under_crates_owned_by_no_package_lints_the_workspace(repo):
    _write(repo, "crates/not-a-package/foo.rs", "\n")
    _commit(repo)
    assert _select(repo)[0] == "workspace"


def test_testdata_edits_are_neutral_but_a_deletion_can_break_include_str(repo):
    _write(repo, "testdata/a.xyz", "edit\n")
    _commit(repo)
    assert _select(repo)[0] == "none"
    (repo / "testdata/a.xyz").unlink()
    _commit(repo)
    assert _select(repo)[0] == "workspace"


def test_uncommitted_and_untracked_crate_files_count(repo):
    # The gate lints the working tree, so the selection has to see it.
    _write(repo, "crates/alpha/src/lib.rs", "// dirty\n")
    _write(repo, "crates/beta/src/new.rs", "// untracked\n")
    assert _select(repo)[:2] == ("crates", ["alpha-pkg", "beta-pkg"])


def test_untracked_scratch_at_the_top_is_ignored_but_an_untracked_lint_config_is_not(
    repo,
):
    _write(repo, "scratch-notes.txt")
    assert _select(repo)[0] == "none"
    _write(repo, "clippy.toml", "# new\n")
    assert _select(repo)[0] == "workspace"


def test_a_rebased_stacked_branch_diffs_against_the_merge_base_only(repo):
    # main moves on after the branch forks: its changes must not be selected.
    _write(repo, "crates/alpha/src/lib.rs", "// branch\n")
    _commit(repo)
    _git(repo, "checkout", "-q", "main")
    _write(repo, "crates/beta/src/lib.rs", "// on main\n")
    _commit(repo)
    _git(repo, "push", "-q", "origin", "main")
    _git(repo, "checkout", "-q", "feature")
    assert _select(repo)[:2] == ("crates", ["alpha-pkg"])
    # Stacked on a second local commit: both still count, the base has not moved.
    _write(repo, "crates/gamma/src/lib.rs", "// stacked\n")
    _commit(repo)
    assert _select(repo)[:2] == ("crates", ["alpha-pkg", "gamma-pkg"])


def test_a_fixed_range_ignores_the_tree_and_does_not_fetch(repo):
    _write(repo, "crates/alpha/src/lib.rs", "// edit\n")
    _commit(repo)
    head = _git(repo, "rev-parse", "HEAD")
    _write(repo, "crates/beta/src/lib.rs", "// dirty, not in the range\n")
    _git(repo, "remote", "set-url", "origin", str(repo.parent / "gone.git"))
    args = ("--base", f"{head}^", "--head", head)
    assert _select(repo, *args)[:2] == ("crates", ["alpha-pkg"])


# ---- the neutral list against the real tree -------------------------------


def _load_selector():
    spec = importlib.util.spec_from_file_location("ci_gate_clippy_scope", SELECTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INCLUDE = re.compile(r'(?:include(?:_str|_bytes)?!\(\s*|#\[path\s*=\s*)"([^"]+)"')


@pytest.mark.parametrize(
    "path", ["rust-toolchain", "rust-toolchain.toml", ".clippy.toml"]
)
def test_toolchain_files_select_the_workspace(path):
    # Direct to classify(): a stub toolchain file would make cargo itself fail.
    result = _load_selector().classify([("M", path)], {"crates/a": "a-pkg"})
    assert result.mode == "workspace" and path in result.reason


def test_no_rust_source_includes_a_path_the_selector_calls_neutral():
    """Derived from the tree: a new include into a neutral path fails here."""
    selector = _load_selector()
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "*.rs"],
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        pytest.skip("not a git checkout")
    sources = tracked.stdout.split()
    assert sources, "found no Rust sources to scan"
    offenders = []
    for rel in sources:
        text = (REPO / rel).read_text(errors="replace")
        for target in INCLUDE.findall(text):
            resolved = os.path.normpath(os.path.join(os.path.dirname(rel), target))
            if selector.is_neutral(resolved):
                offenders.append(f"{rel} -> {resolved}")
    assert not offenders, (
        "a neutral path is now a compile input; take it out of NEUTRAL_* in "
        f"{SELECTOR.name}: {offenders}"
    )


# ---- the real gate, with a stub cargo --------------------------------------

WARNING = json.dumps(
    {
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "unneeded `return` statement",
            "code": {"code": "clippy::needless_return", "explanation": None},
            "spans": [{"is_primary": True, "file_name": "src/lib.rs", "line_start": 1}],
        },
    },
    separators=(",", ":"),
)


def _install_gate(repo, tmp_path):
    """Gate plus lock lib in the clone, on a private lock, with a stub cargo."""
    for name in ("ci-gate.sh", "cargo-lock-lib.sh", SELECTOR.name):
        shutil.copy(SCRIPTS / name, repo / "scripts" / name)
    gate = repo / "scripts" / "ci-gate.sh"
    text = gate.read_text()
    needle = 'LOCK_FILE="/tmp/ferric-cargo.lock"'
    assert needle in text, (
        "the gate's lock path moved; this test must not use the real one"
    )
    gate.write_text(text.replace(needle, f'LOCK_FILE="{tmp_path}/lock"'))

    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    record = tmp_path / "clippy-args"
    stub = stub_dir / "cargo"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'if [[ "$1" == clippy ]]; then\n'
        f'  echo "$*" >> "{record}"\n'
        f'  [[ -e "{tmp_path}/warn" ]] && cat "{tmp_path}/warn"\n'
        "  exit 0\n"
        "fi\n"
        f'exec "{REAL_CARGO}" "$@"\n'
    )
    stub.chmod(0o755)
    (tmp_path / "warn.json").write_text(WARNING + "\n")
    # Commit the scripts so they read as a scripts-only change, then push them
    # to main: the tests below start from a clean base.
    _commit(repo, "gate")
    _git(repo, "push", "-q", "origin", "HEAD:main")
    return gate, record, stub_dir


def _run_gate(gate, stub_dir, **extra):
    env = {
        **_env(),
        "PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}",
        "CI_GATE_SKIP_TESTS": "1",
        "CI_GATE_SKIP_COMPLEXITY": "1",
        "CI_GATE_SKIP_DOCS": "1",
        "CI_GATE_SKIP_PYTEST": "1",
        **extra,
    }
    return subprocess.run(
        [str(gate)], capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL
    )


def test_gate_lints_only_the_changed_crate_and_prints_it(repo, tmp_path):
    gate, record, stub = _install_gate(repo, tmp_path)
    _write(repo, "crates/alpha/src/lib.rs", "// edit\n")
    _commit(repo)
    r = _run_gate(gate, stub)
    assert r.returncode == 0, r.stdout
    args = record.read_text().split()
    assert args[args.index("-p") + 1] == "alpha-pkg"
    assert {"--all-targets", "--no-deps", "--message-format=json"} <= set(args)
    assert "--workspace" not in args
    assert "clippy linted: crates: alpha-pkg" in r.stdout


def test_gate_still_fails_on_a_warning_in_a_scoped_run(repo, tmp_path):
    gate, _, stub = _install_gate(repo, tmp_path)
    _write(repo, "crates/alpha/src/lib.rs", "// edit\n")
    _commit(repo)
    shutil.copy(tmp_path / "warn.json", tmp_path / "warn")
    r = _run_gate(gate, stub)
    assert r.returncode == 1, r.stdout
    assert "clippy::needless_return" in r.stdout
    assert "clippy linted: crates: alpha-pkg" in r.stdout


def test_gate_runs_no_clippy_on_a_docs_only_push_and_says_so(repo, tmp_path):
    gate, record, stub = _install_gate(repo, tmp_path)
    _write(repo, "README.md", "edit\n")
    _commit(repo)
    r = _run_gate(gate, stub)
    assert r.returncode == 0, r.stdout
    assert not record.exists(), "clippy ran on a docs-only push"
    assert "clippy linted: none (" in r.stdout


def test_the_override_forces_the_workspace(repo, tmp_path):
    gate, record, stub = _install_gate(repo, tmp_path)
    _write(repo, "README.md", "edit\n")
    _commit(repo)
    r = _run_gate(gate, stub, CI_GATE_CLIPPY_WORKSPACE="1")
    assert r.returncode == 0, r.stdout
    args = record.read_text().split()
    assert "--workspace" in args and "--no-deps" not in args and "-p" not in args
    assert "clippy linted: workspace (CI_GATE_CLIPPY_WORKSPACE=1)" in r.stdout


@pytest.mark.parametrize(
    "broken",
    [
        "import sys; sys.exit(3)\n",
        "print('garbage')\n",
        "print('crates'); print(''); print('no crates listed')\n",
        "print('workspace')\n",
    ],
    ids=["crash", "garbage", "crates-without-names", "workspace-without-reason"],
)
def test_a_broken_selector_lints_the_workspace(repo, tmp_path, broken):
    gate, record, stub = _install_gate(repo, tmp_path)
    _write(repo, "scripts/" + SELECTOR.name, broken)
    r = _run_gate(gate, stub)
    assert r.returncode == 0, r.stdout
    assert "--workspace" in record.read_text().split()
    assert "clippy linted: workspace (scope selector failed)" in r.stdout


def test_gate_summary_reports_a_skipped_clippy(repo, tmp_path):
    gate, record, stub = _install_gate(repo, tmp_path)
    r = _run_gate(gate, stub, CI_GATE_SKIP_CLIPPY="1")
    assert r.returncode == 0, r.stdout
    assert not record.exists()
    assert "clippy linted: skipped (CI_GATE_SKIP_CLIPPY=1)" in r.stdout
