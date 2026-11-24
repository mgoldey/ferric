"""Basis-aware default r0 of run_terfc_rimp2 and the aTZ-only SCS-MP2(2terfc) fit.

Table: ferric_mp2::terfc_r0 (Goldey & Head-Gordon, Chem. Phys. Lett. 608, 249
(2014); dissertation Tables 6.2 and 6.4). HOW THIS FAILS IF BROKEN: a default
back at 1.05 Angstrom makes the cc-pVDZ default differ from the explicit 1.55
run (and equal the explicit 1.05 run); a silent fallback for an untabulated
basis stops raising ValueError.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ferric = pytest.importorskip("ferric")

H2 = "2\nh2\nH 0 0 0\nH 0 0 0.74\n"
needs_terf = pytest.mark.skipif(
    not (
        os.environ.get("FERRIC_TERF_TABLE_DIR")
        and (Path(os.environ["FERRIC_TERF_TABLE_DIR"]) / "16_4_2.bin").is_file()
    ),
    reason="terfc interpolation tables absent (set FERRIC_TERF_TABLE_DIR)",
)


def _sys(basis, aux):
    return (
        ferric.Molecule.from_xyz_string(H2),
        ferric.BasisSet.bundled(basis),
        ferric.BasisSet.bundled(aux),
    )


@needs_terf
def test_default_r0_is_the_table_value_and_explicit_wins():
    args = _sys("cc-pvdz", "cc-pvdz-ri")
    default = ferric.run_terfc_rimp2(*args).total_energy
    table = ferric.run_terfc_rimp2(*args, r0=1.55).total_energy
    old = ferric.run_terfc_rimp2(*args, r0=1.05).total_energy
    assert abs(default - table) < 1e-10
    assert abs(default - old) > 1e-7


def test_untabulated_basis_without_r0_is_a_value_error():
    args = _sys("sto-3g", "cc-pvdz-ri")
    with pytest.raises(ValueError, match="set r0 explicitly"):
        ferric.run_terfc_rimp2(*args)


def test_scs_2terfc_default_is_the_atz_fit_only():
    args = _sys("cc-pvdz", "cc-pvdz-ri")
    with pytest.raises(ValueError, match="aug-cc-pVTZ only"):
        ferric.run_scs_mp2_2terfc(*args)
    with pytest.raises(ValueError, match="r0_bonded"):
        ferric.run_scs_mp2_2terfc(*args, r0_bonded=0.75)
