"""The QM/MM page must not claim absent what the repo now has.

`site/src/using/qmmm.md` said "no PDB / prmtop / GRO reader, no solvation" and
"not wired into the CLI TOML" for days after all three landed. Nothing checked
it, so the page went stale in the one direction that matters: it told a reader
a capability was missing when it was not.

These assert the NEGATIVE claims against the code, because that is the
direction that rots. A doc understating what exists costs a user the feature;
a doc overstating it costs them a crash, which they at least notice.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PAGE = REPO / "site/src/using/qmmm.md"
README = REPO / "README.md"


@pytest.mark.skipif(not PAGE.is_file(), reason=f"no {PAGE}")
@pytest.mark.parametrize("doc", ["page", "readme"])
def test_no_stale_absence_claims(doc):
    """Phrases that were true once and are not any more."""
    text = (PAGE if doc == "page" else README).read_text()

    # Each entry: the phrase, and what now makes it false.
    forbidden = [
        ("no PDB", "tools/structure reads PDB/mmCIF via gemmi"),
        ("No PDB", "tools/structure reads PDB/mmCIF via gemmi"),
        ("GRO reader", "tools/structure reads GROMACS .gro"),
        ("GRO parsing", "tools/structure reads GROMACS .gro"),
        ("no solvation", "tools/active_site/solvate.py builds a TIP3P droplet"),
        ("No solvation", "tools/active_site/solvate.py builds a TIP3P droplet"),
        ("not wired into the CLI", "config.rs has a [qmmm] TOML section"),
        ("Not wired into the CLI", "config.rs has a [qmmm] TOML section"),
    ]
    hits = [(p, why) for p, why in forbidden if p in text]
    assert not hits, f"{doc} claims something is absent that now exists: " + "; ".join(
        f"{p!r} -- but {why}" for p, why in hits
    )


@pytest.mark.skipif(not PAGE.is_file(), reason=f"no {PAGE}")
def test_the_capabilities_the_page_claims_actually_exist():
    """The positive direction, spot-checked against the code.

    Cheap string checks against the SOURCE, not imports: this runs in the fast
    tier and must not need rdkit, a receptor or a compiled extension.
    """
    text = PAGE.read_text()

    checks = [
        (".gro", REPO / "tools/structure/__init__.py", '".gro": "gro"'),
        ("solvation droplet", REPO / "tools/active_site/solvate.py", "def solvate("),
        (
            "[qmmm]",
            REPO / "crates/ferric-cli/src/config.rs",
            "pub qmmm: Option<QmmmCfg>",
        ),
    ]
    for phrase, path, needle in checks:
        if phrase not in text:
            continue  # the page may legitimately stop mentioning it
        assert path.is_file(), f"page mentions {phrase!r} but {path} is gone"
        assert needle in path.read_text(), (
            f"page mentions {phrase!r} but {path.name} no longer contains "
            f"{needle!r} -- the capability was removed and the page still "
            f"advertises it"
        )


@pytest.mark.skipif(not PAGE.is_file(), reason=f"no {PAGE}")
def test_the_page_is_about_the_SOFTWARE_not_the_repo():
    """No meta-commentary about the documentation's own history.

    The page opened with "until this page it was mentioned nowhere in the
    README" and "an external reviewer reading the repo concluded...". That is
    project history; a reader wants to know what the software does.
    """
    text = PAGE.read_text().lower()
    meta = [
        "until this page",
        "previously undocumented",
        "an external reviewer",
        "was invisible",
        "nowhere in the readme",
        "the documentation was not",
    ]
    hits = [m for m in meta if m in text]
    assert not hits, (
        f"the QM/MM page carries repo history rather than user-facing "
        f"documentation: {hits}"
    )


@pytest.mark.skipif(not PAGE.is_file(), reason=f"no {PAGE}")
def test_every_toml_key_the_page_shows_is_real():
    """A key that does not exist is a config the reader cannot run.

    `[qmmm]` is `deny_unknown_fields`, so a typo'd key in the docs is a hard
    error for whoever pastes it.
    """
    config = REPO / "crates/ferric-cli/src/config.rs"
    if not config.is_file():  # pragma: no cover - source checkout only
        pytest.skip("ferric-cli source not present")

    block = re.search(r"```toml\n(.*?)```", PAGE.read_text(), re.S)
    assert block, "the page no longer shows a TOML block; re-derive this guard"
    # The block is a complete config ([molecule], [basis], [method], [qmmm]);
    # only the [qmmm] table's keys belong to QmmmCfg. Commented-out keys in
    # that table are still offered to the reader, so they are checked too.
    documented = set()
    section = None
    for line in block.group(1).splitlines():
        header = re.match(r"\s*\[([^\]]+)\]\s*$", line)
        if header:
            section = header.group(1).strip()
            continue
        if section == "qmmm" and "=" in line:
            documented.add(line.split("=")[0].strip().lstrip("#").strip())
    assert documented, "no [qmmm] keys parsed from the TOML block"

    src = config.read_text()
    start = src.index("pub struct QmmmCfg")
    real = {
        m.group(1)
        for m in re.finditer(
            r"\n\s*pub\s+(\w+)\s*:", src[start : src.index("\n}", start)]
        )
    }
    unknown = documented - real
    assert not unknown, (
        f"the page's [qmmm] block names key(s) that QmmmCfg does not have: "
        f"{sorted(unknown)}. The section is deny_unknown_fields, so pasting "
        f"the example would hard-error."
    )
