"""The ``gc_kernel`` keyword of the six molecular SCF entry points.

It sets ``RhfConfig::gc_kernel`` (the CLI's ``[scf] gc_kernel``): the
shared-primitive ``(ss|ss)`` / ``(ss|sp)`` kernel of the contraction-rotated
direct J/K walk. ``None`` is ``"off"`` (the default, ``FERRIC_GC_KERNEL``
unset) bit for bit; ``"auto"`` agrees with it to rounding (the Rust tests in
``crates/ferric-scf/tests/gc_kernel.rs`` hold the J/K-level bars); a basis
with nothing to rotate (def2-SVP), or ``gc_rotation="off"``, is unchanged
bit for bit; any other value (other cases and padded spellings included)
raises.
"""

from __future__ import annotations

import os

import ferric
import pytest

TIGHT = {"energy_conv": 1e-10, "density_conv": 1e-8}
EXACT = {"df_j_aux": "", "df_k_aux": "", "eri": "direct"}
WATER = "3\nwater\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n"
NH2 = "3\nNH2\nN 0 0 0.15\nH 0 0.80 -0.45\nH 0 -0.80 -0.45\n"
ENV_UNSET = os.environ.get("FERRIC_GC_KERNEL") is None


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
def test_gc_kernel_values_agree_and_bad_value_raises(name):
    run, extra = entry(name)
    mol = mol_for(name)
    bs = ferric.BasisSet.bundled("cc-pvdz")
    e = {
        v: run(mol, bs, gc_kernel=v, **extra, **EXACT, **TIGHT).energy
        for v in ("off", "auto", None)
    }
    assert abs(e["auto"] - e["off"]) <= 1e-10, e
    if ENV_UNSET:
        assert e[None] == e["off"], e
    for bad in ("on", "OFF", "Auto", " auto"):
        with pytest.raises(ValueError, match="gc_kernel"):
            run(mol, bs, gc_kernel=bad, **extra, **EXACT, **TIGHT)


@pytest.mark.parametrize("name", ["rhf", "uhf"])
def test_kernel_is_inert_where_it_does_not_apply(name):
    run, extra = entry(name)
    mol = mol_for(name)
    svp = ferric.BasisSet.bundled("def2-svp")
    e = [
        run(mol, svp, gc_kernel=v, **extra, **EXACT, **TIGHT).energy
        for v in ("off", "auto")
    ]
    assert e[0] == e[1], e
    dz = ferric.BasisSet.bundled("cc-pvdz")
    e = [
        run(mol, dz, gc_rotation="off", gc_kernel=v, **extra, **EXACT, **TIGHT).energy
        for v in ("off", "auto")
    ]
    assert e[0] == e[1], e
