"""`python -m tools.tox` — the entry point the library never had.

A screen that cannot be run from a shell gets used by the code that already
imports it and by nobody else.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("rdkit", reason="the local alert screen is RDKit-based")

from tools.tox.__main__ import main  # noqa: E402

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def test_a_known_molecule_reports_its_descriptors(capsys):
    assert main(["--offline", ASPIRIN]) == 0
    out = capsys.readouterr().out
    # Aspirin: MW 180.2, TPSA 63.6. Assert the VALUES, not just the labels --
    # a run that printed every endpoint name with no numbers would otherwise
    # pass.
    assert "180.2" in out
    assert "63.6" in out
    assert "desc_mw" in out


def test_polarity_is_stated_for_every_endpoint(capsys):
    """A bare number invites 'higher is worse', which is false for half."""
    main(["--offline", ASPIRIN])
    out = capsys.readouterr().out
    # Endpoint lines start with exactly two spaces; notes are indented six.
    body = [
        ln
        for ln in out.splitlines()
        if ln.startswith("  ") and not ln.startswith("      ") and "rdkit-alerts" in ln
    ]
    assert len(body) >= 5, f"expected several endpoint lines, got {len(body)}"
    missing = [ln for ln in body if "higher=" not in ln]
    assert not missing, (
        f"{len(missing)} endpoint lines carry no polarity: {missing[:2]}"
    )


def test_json_is_machine_readable(capsys):
    assert main(["--offline", "--json", "CCO"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data) == 1
    assert data[0]["smiles"] == "CCO"
    assert any(e["name"] == "desc_mw" for e in data[0]["endpoints"])
    # Polarity must survive into JSON, or an aggregator inverts a ranking.
    assert all("higher_is_worse" in e for e in data[0]["endpoints"])


def test_an_unparseable_smiles_EXITS_NONZERO(capsys):
    """Silently scoring nothing as clean is the failure being prevented."""
    assert main(["--offline", "not!a!smiles"]) == 1
    assert "could not assess" in capsys.readouterr().err


def test_a_smi_file_is_read_with_its_labels(tmp_path, capsys):
    f = tmp_path / "set.smi"
    f.write_text(f"{ASPIRIN} aspirin\n# a comment\nCCO ethanol\n")
    assert main(["--offline", str(f)]) == 0
    out = capsys.readouterr().out
    assert "aspirin" in out and "ethanol" in out
    assert "# a comment" not in out


def test_multiple_molecules_are_all_reported(capsys):
    assert main(["--offline", ASPIRIN, "CCO"]) == 0
    out = capsys.readouterr().out
    assert out.count("desc_mw") == 2


def test_offline_makes_no_network_call(monkeypatch, capsys):
    """--offline must not reach a web provider even if one is reachable."""
    import tools.tox.web as web

    def explode(*a, **k):
        raise AssertionError("--offline made a network call")

    monkeypatch.setattr(web, "_post_json", explode)
    assert main(["--offline", ASPIRIN]) == 0
