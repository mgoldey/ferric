"""The Kohn-Sham geometry drivers take the CLI's J/K default for every spin.

`run_optimize` and `run_frequencies` with `xc=` and no `df_j_aux` /
`df_k_aux` fit J and K on def2-universal-jkfit for RKS, UKS and ROKS, as
`run_dft` / `run_uks` / `run_roks` and the CLI's `task = "optimize"` do. HF
(no `xc`) stays four-centre.

Each check compares the first SCF energy of a one-step optimization (the
start geometry, so no step is taken in between) against explicit routes:
equal to the explicit def2 run (the same route, bitwise) and different from
`"none"` (four-centre J/K). The RI-J error of OH/STO-3G on that aux basis is
~1e-4 Ha, so "different" is far above round-off. A route that ignored the
default would equal the `"none"` run instead.
"""

from __future__ import annotations

import pytest

ferric = pytest.importorskip("ferric")

AUX = "def2-universal-jkfit"
OH = "2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n"
WATER = "3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n"


def first_energy(mol, **kw):
    r = ferric.run_optimize(mol, "sto-3g", max_steps=1, **kw)
    return r.energy_trace[0]


@pytest.mark.parametrize(
    ("xyz", "mult", "reference", "xc"),
    [
        (OH, 2, "uhf", "PBE0"),
        (OH, 2, "rohf", "PBE0"),
        (WATER, 1, "rhf", "PBE0"),
    ],
)
def test_ks_optimize_defaults_to_ri_jk_for_every_reference(xyz, mult, reference, xc):
    mol = ferric.Molecule.from_xyz_string(xyz, multiplicity=mult)
    kw = {"reference": reference, "xc": xc}
    unset = first_energy(mol, **kw)
    assert unset == first_energy(mol, df_j_aux=AUX, df_k_aux=AUX, **kw)
    exact = first_energy(mol, df_j_aux="none", df_k_aux="none", **kw)
    assert abs(unset - exact) > 1e-6, (
        f"{reference}: the unset route equals four-centre J/K ({unset} vs {exact})"
    )


def test_hf_optimize_stays_four_centre():
    mol = ferric.Molecule.from_xyz_string(OH, multiplicity=2)
    unset = first_energy(mol, reference="uhf")
    assert unset == first_energy(mol, reference="uhf", df_j_aux="none", df_k_aux="none")


def test_uks_frequencies_default_to_ri_jk():
    """`run_frequencies` reads the same rule: its reference energy is the fitted SCF."""
    mol = ferric.Molecule.from_xyz_string(OH, multiplicity=2)
    kw = {"reference": "uhf", "xc": "PBE", "hessian": "fd"}
    unset = ferric.run_frequencies(mol, "sto-3g", **kw)
    explicit = ferric.run_frequencies(mol, "sto-3g", df_j_aux=AUX, df_k_aux=AUX, **kw)
    assert unset.energy == explicit.energy
    assert unset.frequencies == explicit.frequencies
    exact = ferric.run_frequencies(
        mol, "sto-3g", df_j_aux="none", df_k_aux="none", **kw
    )
    assert abs(unset.energy - exact.energy) > 1e-6
