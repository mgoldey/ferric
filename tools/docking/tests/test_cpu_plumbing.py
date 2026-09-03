"""`cpu` must reach Vina, because its default silently takes the whole box.

Vina's `cpu=0` means "every core". That is right for docking one ligand and
wrong for a screen: Vina parallelizes across internal search runs and cannot
fill all cores at low exhaustiveness (it says so itself), so a sequential loop
over N ligands leaves the machine partly idle N times. Screening wants
`cpu=1` per ligand with fan-out ACROSS ligands.

These tests use a fake Vina so they cost nothing and run without the optional
`docking` extra installed.
"""
from __future__ import annotations

import sys
import types

import pytest

from tools.docking import vina_dock


class _FakeVina:
    """Records constructor kwargs; produces one trivial PDBQT model."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        type(self).last_kwargs = dict(kwargs)

    def set_receptor(self, *a, **k): pass
    def set_ligand_from_string(self, *a, **k): pass
    def compute_vina_maps(self, *a, **k): pass
    def dock(self, *a, **k): pass

    def poses(self, n_poses=1):
        return (
            "MODEL 1\n"
            "REMARK VINA RESULT:    -7.5      0.000      0.000\n"
            "ATOM      1  C   LIG A   1       0.000   0.000   0.000  "
            "0.00  0.00     0.000 C \n"
            "ENDMDL\n"
        )


@pytest.fixture
def fake_vina(monkeypatch, tmp_path):
    mod = types.ModuleType("vina")
    mod.Vina = _FakeVina
    monkeypatch.setitem(sys.modules, "vina", mod)
    monkeypatch.setattr(vina_dock, "_ligand_pdbqt_from_rdkit",
                        lambda mol: "LIGAND PDBQT")
    receptor = tmp_path / "r.pdbqt"
    receptor.write_text("ATOM\n")
    return receptor


def test_cpu_defaults_to_zero_meaning_all_cores(fake_vina):
    """Preserves Vina's own default for the single-ligand case."""
    vina_dock.dock_ligand(object(), fake_vina, (0.0, 0.0, 0.0))
    assert _FakeVina.last_kwargs["cpu"] == 0


def test_cpu_is_forwarded_to_vina(fake_vina):
    """The whole point: a screen must be able to say 'one core per ligand'."""
    vina_dock.dock_ligand(object(), fake_vina, (0.0, 0.0, 0.0), cpu=1)
    assert _FakeVina.last_kwargs["cpu"] == 1


def test_seed_is_still_forwarded_alongside_cpu(fake_vina):
    """Adding cpu must not displace the seed -- both are needed to reproduce.

    Vina's threads race to fill the pose buffer, so the result depends on
    thread count as well as seed. Pinning one without the other does not make
    a screen reproducible.
    """
    vina_dock.dock_ligand(object(), fake_vina, (0.0, 0.0, 0.0),
                          seed=1234, cpu=2)
    assert _FakeVina.last_kwargs["seed"] == 1234
    assert _FakeVina.last_kwargs["cpu"] == 2


def test_docstring_warns_that_cpu_zero_takes_the_box(fake_vina):
    """The trap is silent, so the warning must stay in the docs."""
    doc = vina_dock.dock_ligand.__doc__ or ""
    assert "every core" in doc
    assert "cpu=1" in doc
