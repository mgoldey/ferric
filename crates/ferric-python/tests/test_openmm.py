"""Tests for ``ferric.openmm`` (OpenMM -> ``MmTopology`` parameters)."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

openmm = pytest.importorskip(
    "openmm",
    reason="OpenMM not installed",
)

ferric = pytest.importorskip("ferric")

FIXTURE = REPO / "testdata/molecules/validation/mm/ala_ala.pdb"


def test_ferric_openmm_reexports_and_runs():
    import ferric.openmm as om

    result = om.topology_from_openmm(FIXTURE)
    assert result["n_atoms"] == 23
    assert len(result["charges"]) == 23

    topo = ferric.MmTopology.from_amber_units(
        result["charges"],
        result["sigmas_angstrom"],
        result["epsilons_kcal"],
        result["bonds"],
        result["angles"],
        result["torsions"],
    )
    assert topo.n_atoms() == 23
