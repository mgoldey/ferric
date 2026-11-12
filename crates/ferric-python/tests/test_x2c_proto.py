"""AoIntegrals.pvp and the spin-free X2C-1e prototype (scripts/prototypes/x2c_proto.py).

Ne/cc-pVDZ only, to keep the suite fast; the prototype's own run covers Ar,
Kr and HBr. Tolerances are the prototype's derived bars.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ferric = pytest.importorskip("ferric")
if not hasattr(ferric.AoIntegrals, "pvp"):
    pytest.skip("loaded ferric build has no AoIntegrals.pvp", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "x2c_proto", ROOT / "scripts/prototypes/x2c_proto.py"
)
x2c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(x2c)


@pytest.fixture(scope="module")
def neon(tmp_path_factory):
    return x2c.System("Ne", tmp=tmp_path_factory.mktemp("x2c"))


def test_pvp_is_symmetric(neon):
    w = neon.w_u
    assert np.abs(w - w.T).max() <= x2c.ROUND * np.abs(w).max()


def test_contraction_matrix_maps_s_t_v(neon):
    assert x2c.anchor_contract(neon)


def test_zero_hook_is_bitwise_the_plain_run(neon):
    assert x2c.anchor_hook0(neon, energy_conv=1e-10, density_conv=1e-8)


def test_correction_falls_as_c_minus_two(neon):
    pytest.importorskip("pyscf")
    assert x2c.anchor_cscan(neon)


def test_pvp_eigenvalues_match_pyscf(neon):
    pytest.importorskip("pyscf")
    assert x2c.anchor_pvp(neon)


def test_x2c_hcore_and_energy_shift_match_pyscf(neon):
    pytest.importorskip("pyscf")
    assert x2c.anchor_hcore(neon)
    assert x2c.anchor_scf(neon, 1e-10)


def test_flipped_w_passes_the_c_limit_but_not_pyscf(neon):
    pytest.importorskip("pyscf")
    assert x2c.anchor_cscan(neon, flip_w=True)
    assert not x2c.anchor_hcore(neon, flip_w=True)
