"""The reusable workflows ferric's CI calls from mgoldey/attestation (roadmap 39) are pinned, and pinned together.

A called workflow runs with this repository's token and checkout, so a moving
ref (a branch or a tag) would let a change in another repository alter what
ferric's CI runs. Every `uses:` of an attestation reusable workflow must name a
full 40-hex commit SHA, and all of them the SAME one: the templates are one
release of attestation's CI. Also: python-quality's ruff pin stays equal to
.pre-commit-config.yaml's ruff rev (ci.yml's VERSION PINNING TRAP comment).
Read as text, no yaml dependency. Extension-free (conftest.EXTENSION_FREE).
Spec: attestation's docs/superpowers/specs/2026-11-20-shared-ci-templates-design.md.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO / ".github" / "workflows"
REF = re.compile(
    r"^\s*uses:\s*mgoldey/attestation/\.github/workflows/(reusable-[\w-]+\.yml)@(\S+)",
    re.MULTILINE,
)


def _refs():
    return [
        (p.name, m[1], m[2])
        for p in sorted(WORKFLOWS.glob("*.yml"))
        for m in REF.finditer(p.read_text(encoding="utf-8"))
    ]


def test_attestation_reusable_workflows_are_full_sha_pinned_to_one_commit():
    refs = _refs()
    assert refs, "no workflow calls attestation's reusable workflows"
    for wf, template, ref in refs:
        assert re.fullmatch(r"[0-9a-f]{40}", ref), (
            f"{wf}: {template}@{ref} is not a full commit SHA"
        )
    assert len({ref for *_, ref in refs}) == 1, f"pinned to different SHAs: {refs}"


def test_only_pre_commit_and_python_quality_are_called():
    assert {(wf, t) for wf, t, _ in _refs()} == {
        ("ci.yml", "reusable-pre-commit.yml"),
        ("ci.yml", "reusable-python-gates.yml"),
    }


def test_python_quality_ruff_pin_matches_pre_commit_rev():
    ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    pins = re.findall(r"^\s*ruff-version:\s*\"?([^\s\"#]+)\"?", ci, re.MULTILINE)
    assert len(pins) == 1, pins
    hooks = (REPO / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    rev = re.search(
        r"repo: https://github.com/astral-sh/ruff-pre-commit.*?rev:\s*v(\S+)",
        hooks,
        re.DOTALL,
    )
    assert rev and rev[1] == pins[0], (rev and rev[1], pins[0])
