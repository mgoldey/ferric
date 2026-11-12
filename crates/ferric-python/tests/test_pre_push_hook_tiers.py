"""The pre-push hook's tiers set the gate knobs their banners promise.

`scripts/install-hooks.sh` writes the pre-push hook from a heredoc. The hook
picks a tier (CI_GATE_TIER, default `lint`) and exports CI_GATE_* knobs for
`scripts/ci-gate.sh`. The test step and the soft pytest step are gated by
SEPARATE knobs (CI_GATE_SKIP_TESTS, CI_GATE_SKIP_PYTEST).

MEASURED 2026-10-29: the lint tier set only CI_GATE_SKIP_TESTS while its
banner said "Tests are NOT run here", so every push ran the whole binding
suite: 34 min, then 55 min under load, advisory (it could not block the
push), duplicating CI, and holding target/release/libferric.so open so no
rebuild could start. With CI_GATE_SKIP_PYTEST in the lint tier the same push
took 11 s.

The hook is DERIVED from install-hooks.sh (its heredoc), not hand-copied, and
run against a stub gate that prints the knobs it received, so a tier that
drifts from its banner fails here.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

# The hook is a `#!/usr/bin/env bash` script executed directly; Windows cannot
# exec a shebang file (WinError 193), and git hooks there run under Git's sh.
pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="executes a shebang script directly"
)

REPO = Path(__file__).resolve().parents[3]
INSTALLER = REPO / "scripts" / "install-hooks.sh"
KNOBS = ("CI_GATE_SKIP_TESTS", "CI_GATE_SKIP_PYTEST", "CI_GATE_FAST")


def _hook_body():
    m = re.search(r"<<'HOOK_EOF'\n(.*?)\nHOOK_EOF\n", INSTALLER.read_text(), re.DOTALL)
    assert m, "install-hooks.sh no longer writes the hook from a HOOK_EOF heredoc"
    return m.group(1)


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A one-commit repo whose scripts/ci-gate.sh prints the knobs it got."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    gate = tmp_path / "scripts" / "ci-gate.sh"
    gate.parent.mkdir()
    gate.write_text(
        "#!/usr/bin/env bash\n"
        + "".join(f'echo "{k}=${{{k}:-unset}}"\n' for k in KNOBS)
    )
    gate.chmod(0o755)
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    hook = tmp_path / "pre-push"
    hook.write_text(_hook_body() + "\n")
    hook.chmod(0o755)
    return tmp_path


def _push(repo, tier=None, sha=None):
    head = _git(repo, "rev-parse", "HEAD")
    env = {k: v for k, v in os.environ.items() if not k.startswith("CI_GATE_")}
    if tier is not None:
        env["CI_GATE_TIER"] = tier
    stdin = f"refs/heads/main {sha or head} refs/heads/main {'0' * 40}\n"
    return subprocess.run(
        [str(repo / "pre-push")],
        cwd=repo,
        input=stdin,
        env=env,
        capture_output=True,
        text=True,
    )


def _knobs(out):
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


@pytest.mark.parametrize(
    "tier, expected",
    [
        (
            None,
            {
                "CI_GATE_SKIP_TESTS": "1",
                "CI_GATE_SKIP_PYTEST": "1",
                "CI_GATE_FAST": "1",
            },
        ),
        (
            "lint",
            {
                "CI_GATE_SKIP_TESTS": "1",
                "CI_GATE_SKIP_PYTEST": "1",
                "CI_GATE_FAST": "1",
            },
        ),
        (
            "fast",
            {
                "CI_GATE_SKIP_TESTS": "unset",
                "CI_GATE_SKIP_PYTEST": "unset",
                "CI_GATE_FAST": "1",
            },
        ),
        (
            "full",
            {
                "CI_GATE_SKIP_TESTS": "unset",
                "CI_GATE_SKIP_PYTEST": "unset",
                "CI_GATE_FAST": "0",
            },
        ),
    ],
    ids=["default", "lint", "fast", "full"],
)
def test_each_tier_exports_what_it_promises(repo, tier, expected):
    r = _push(repo, tier)
    assert r.returncode == 0, r.stderr
    assert _knobs(r.stdout) == expected
    if expected["CI_GATE_SKIP_PYTEST"] == "1":
        assert "pytest" in r.stderr and "NOT run here" in r.stderr


def test_unknown_tier_and_unchecked_out_ref_are_refused(repo):
    r = _push(repo, "everything")
    assert r.returncode == 1 and "unknown CI_GATE_TIER" in r.stderr
    r = _push(repo, sha="1" * 40)
    assert r.returncode == 1 and "REFUSING" in r.stderr
    assert "CI_GATE_FAST" not in r.stdout  # the gate never ran


# ---- the rustfmt check runs in the lint tier -------------------------------
# The real scripts/ci-gate.sh runs with the knobs the hook's lint tier exports
# (read back from the stub gate above, not hand-listed) and a fake `cargo` that
# logs its arguments. Every step that would take the cargo lock is skipped, so
# the test never queues behind a build on the box.

GATE = REPO / "scripts" / "ci-gate.sh"
LOCKED_STEPS_OFF = {
    "CI_GATE_SKIP_CLIPPY": "1",
    "CI_GATE_SKIP_COMPLEXITY": "1",
    "CI_GATE_SKIP_DOCS": "1",
}


def _run_gate(tmp_path, repo, fmt_rc, extra=None):
    lint = _knobs(_push(repo, "lint").stdout)
    shim = tmp_path / "shim"
    shim.mkdir()
    log = tmp_path / "cargo.log"
    cargo = shim / "cargo"
    cargo.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$*" >> "{log}"\n'
        'if [[ "$1" == fmt ]]; then echo "Diff in src/x.rs at line 1:"; '
        f"exit {fmt_rc}; fi\n"
        "exit 0\n"
    )
    cargo.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("CI_GATE_")}
    env.update(lint)
    env.update(LOCKED_STEPS_OFF)
    env.update(extra or {})
    env["FERRIC_SHIM_DIR"] = str(shim)
    env["PATH"] = f"{shim}:{env['PATH']}"
    r = subprocess.run(
        ["bash", str(GATE)], env=env, capture_output=True, text=True, timeout=120
    )
    calls = log.read_text().splitlines() if log.exists() else []
    return r, calls


def test_lint_tier_runs_cargo_fmt_check(tmp_path, repo):
    r, calls = _run_gate(tmp_path, repo, fmt_rc=0)
    assert r.returncode == 0, r.stdout + r.stderr
    assert calls == ["fmt --all --check"]
    assert "-- cargo fmt: PASS --" in r.stdout
    assert "RESULT: PASS" in r.stdout


def test_a_formatting_diff_fails_the_lint_tier(tmp_path, repo):
    r, calls = _run_gate(tmp_path, repo, fmt_rc=1)
    assert calls == ["fmt --all --check"]
    assert r.returncode == 1, r.stdout
    assert "Diff in src/x.rs" in r.stdout
    assert "-- cargo fmt: FAIL (exit 1)" in r.stdout
    assert "RESULT: FAIL" in r.stdout


def test_fmt_check_can_be_skipped(tmp_path, repo):
    r, calls = _run_gate(tmp_path, repo, fmt_rc=1, extra={"CI_GATE_SKIP_FMT": "1"})
    assert r.returncode == 0, r.stdout
    assert calls == []
    assert "-- cargo fmt: SKIPPED (CI_GATE_SKIP_FMT=1) --" in r.stdout
