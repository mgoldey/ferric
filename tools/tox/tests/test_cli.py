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


def test_a_smi_file_is_read_as_utf8_explicitly(tmp_path, monkeypatch):
    """The read must not depend on the host's locale encoding.

    `Path.read_text()` with no `encoding=` decodes using the host locale, so a
    valid UTF-8 file with a non-ASCII label can fail on a non-UTF-8 locale.
    Monkeypatching `locale.getpreferredencoding` does not reliably change what
    `read_text()` resolves to on every platform/version, so this asserts the
    call site passes `encoding="utf-8"` explicitly rather than relying on the
    ambient locale to happen to be UTF-8 (as it is on this box).
    """
    import pathlib

    seen: dict[str, object] = {}
    real_read_text = pathlib.Path.read_text

    def spy(self, *args, **kwargs):
        if self.name == "set.smi":
            seen["encoding"] = kwargs.get("encoding") or (args[0] if args else None)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", spy)

    f = tmp_path / "set.smi"
    f.write_bytes(f"{ASPIRIN} café\n".encode("utf-8"))
    from tools.tox.__main__ import _read_inputs

    result = _read_inputs([str(f)])
    assert seen.get("encoding") == "utf-8", (
        f"read_text was called with encoding={seen.get('encoding')!r}; "
        "a locale-dependent read can garble a valid UTF-8 label"
    )
    assert "café" in result


def test_a_duplicate_label_is_an_input_error_not_a_silent_drop(tmp_path, capsys):
    """Two lines sharing a label must not let dict assignment eat the first.

    Silently overwriting means `assess_many` only ever sees the LAST SMILES
    for that label -- the CLI could exit 0 having assessed fewer molecules
    than were requested.
    """
    f = tmp_path / "set.smi"
    f.write_text(f"{ASPIRIN} dup\nCCO dup\n")
    rc = main(["--offline", str(f)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "dup" in err


def test_a_repeated_identical_label_and_smiles_is_not_an_error(tmp_path, capsys):
    """The SAME molecule listed twice under the SAME label is a harmless no-op."""
    f = tmp_path / "set.smi"
    f.write_text(f"{ASPIRIN} aspirin\n{ASPIRIN} aspirin\n")
    assert main(["--offline", str(f)]) == 0
