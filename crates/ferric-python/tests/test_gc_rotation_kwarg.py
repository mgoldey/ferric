"""The ``gc_rotation`` keyword of the six molecular SCF entry points.

It sets ``RhfConfig::gc_rotation`` (the CLI's ``[scf] gc_rotation``): the
internal contraction rotation of generally contracted bases in the exact
four-centre J/K builds. ``"off"`` is the unrotated build, ``"auto"`` /
``"rot1"`` agree with it to the screening noise (the Rust tests in
``crates/ferric-scf/tests/gc_rotation.rs`` hold the J/K-level bars), a basis
with nothing to rotate is unchanged bit for bit, and any other value (other
cases and padded spellings included) raises.
"""

from __future__ import annotations

import ferric
import pytest

TIGHT = {"energy_conv": 1e-10, "density_conv": 1e-8}
EXACT = {"df_j_aux": "", "df_k_aux": "", "eri": "direct"}
WATER = "3\nwater\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n"
NH2 = "3\nNH2\nN 0 0 0.15\nH 0 0.80 -0.45\nH 0 -0.80 -0.45\n"


def entry(name):
    return {
        "rhf": (ferric.run_rhf, {}),
        "uhf": (ferric.run_uhf, {}),
        "rohf": (ferric.run_rohf, {}),
        "dft": (ferric.run_dft, {"functional": "PBE0"}),
        "uks": (ferric.run_uks, {"functional": "PBE0"}),
        "roks": (ferric.run_roks, {"functional": "PBE0"}),
    }[name]


def mol_for(name):
    if name in ("rhf", "dft"):
        return ferric.Molecule.from_xyz_string(WATER)
    return ferric.Molecule.from_xyz_string(NH2, charge=0, multiplicity=2)


@pytest.mark.parametrize("name", ["rhf", "uhf", "rohf", "dft", "uks", "roks"])
def test_gc_rotation_values_agree_and_bad_value_raises(name):
    run, extra = entry(name)
    mol = mol_for(name)
    bs = ferric.BasisSet.bundled("cc-pvdz")
    e = {
        v: run(mol, bs, gc_rotation=v, **extra, **EXACT, **TIGHT).energy
        for v in ("off", "auto", "rot1")
    }
    assert abs(e["auto"] - e["off"]) <= 1e-10, e
    assert abs(e["rot1"] - e["off"]) <= 1e-10, e
    # The spelling is exact, as for [scf] gc_rotation.
    for bad in ("on", "OFF", "Auto", " auto"):
        with pytest.raises(ValueError, match="gc_rotation"):
            run(mol, bs, gc_rotation=bad, **extra, **EXACT, **TIGHT)


@pytest.mark.parametrize("name", ["rhf", "uhf"])
def test_segmented_basis_is_bitwise_unchanged(name):
    run, extra = entry(name)
    mol = mol_for(name)
    bs = ferric.BasisSet.bundled("def2-svp")
    e = [
        run(mol, bs, gc_rotation=v, **extra, **EXACT, **TIGHT).energy
        for v in ("off", "auto", None)
    ]
    assert e[0] == e[1] == e[2], e
