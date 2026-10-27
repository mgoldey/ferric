"""`run_ccsd(..., ladder=)` / `run_ccsd_t(..., ladder=)` reach the library route.

The dense and RI particle-particle ladders are the same algebra over the same
RI tensors (the Rust tests pin them to <1e-10 in E, t1 and t2), so every
spelling must give the same energy. That alone would also pass if the kwarg
were silently ignored, so the RI route is additionally checked to RUN where
the dense one is refused: under a budget the dense plan does not fit,
``ladder="dense"`` raises and ``ladder="ri"`` / ``"auto"`` succeed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ferric = pytest.importorskip("ferric")

WATER = Path(__file__).resolve().parents[3] / "testdata" / "molecules" / "water.xyz"


@pytest.fixture(scope="module")
def water():
    mol = ferric.Molecule.from_xyz(str(WATER))
    return (
        mol,
        ferric.BasisSet.bundled("cc-pvdz"),
        ferric.BasisSet.bundled("cc-pvdz-ri"),
    )


def test_every_ladder_spelling_gives_the_same_ccsd_energy(water):
    mol, obs, aux = water
    r = {
        lad: ferric.run_ccsd(mol, obs, aux, ladder=lad)
        for lad in ("auto", "dense", "ri")
    }
    e = {lad: res.correlation_energy for lad, res in r.items()}
    assert abs(e["auto"]) > 0.1
    assert abs(e["dense"] - e["ri"]) < 1e-10, e
    assert abs(e["auto"] - e["dense"]) < 1e-10, e
    assert (r["dense"].ladder_route, r["dense"].ladder_block) == ("dense", None)
    assert r["auto"].ladder_route == "dense"
    assert r["ri"].ladder_route == "ri" and r["ri"].ladder_block == 19


def test_ri_runs_under_a_budget_the_dense_ladder_is_refused(water):
    mol, obs, aux = water
    # water/cc-pVDZ (nv=19): the dense plan needs ~6.9 MB (vvvv plus the
    # three nv^4 build temporaries), the RI plan 3.4 MB at block 1 and
    # 4.4 MB at full width. 5 MB separates them.
    gb = 5e-3
    with pytest.raises(RuntimeError, match="budget is"):
        ferric.run_ccsd(mol, obs, aux, memory_budget_gb=gb, ladder="dense")
    e_ri = ferric.run_ccsd(
        mol, obs, aux, memory_budget_gb=gb, ladder="ri"
    ).correlation_energy
    auto = ferric.run_ccsd(mol, obs, aux, memory_budget_gb=gb)
    assert auto.ladder_route == "ri", "auto must fall back to RI under this budget"
    e_auto = auto.correlation_energy
    e_dense = ferric.run_ccsd(mol, obs, aux, ladder="dense").correlation_energy
    assert abs(e_ri - e_dense) < 1e-10
    assert abs(e_auto - e_dense) < 1e-10


def test_ccsd_t_takes_the_kwarg(water):
    mol, obs, aux = water
    d = ferric.run_ccsd_t(mol, obs, aux, ladder="dense")
    r = ferric.run_ccsd_t(mol, obs, aux, ladder="ri")
    assert abs(d.correlation_energy - r.correlation_energy) < 1e-10
    assert abs(d.t_correction - r.t_correction) < 1e-10


@pytest.mark.parametrize("bad", ["", "Dense", "RI", "direct"])
def test_unknown_ladder_is_a_value_error(water, bad):
    mol, obs, aux = water
    with pytest.raises(ValueError, match="ladder"):
        ferric.run_ccsd(mol, obs, aux, ladder=bad)
